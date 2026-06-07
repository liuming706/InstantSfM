import numpy as np
import pycolmap
import pypose as pp
import pytest
import torch

from instantsfm.scene.defs import CameraModelId, Cameras, get_camera_model_info
from instantsfm.utils.cost_function import reproject_funcs, reproject_funcs_no_depth


MODEL_CASES = {
    "SIMPLE_PINHOLE": (CameraModelId.SIMPLE_PINHOLE, [800, 500, 400]),
    "PINHOLE": (CameraModelId.PINHOLE, [810, 790, 500, 400]),
    "SIMPLE_RADIAL": (CameraModelId.SIMPLE_RADIAL, [800, 500, 400, 0.01]),
    "RADIAL": (CameraModelId.RADIAL, [800, 500, 400, 0.01, -0.001]),
    "OPENCV": (CameraModelId.OPENCV, [810, 790, 500, 400, 0.01, -0.001, 0.0005, -0.0003]),
    "OPENCV_FISHEYE": (CameraModelId.OPENCV_FISHEYE, [810, 790, 500, 400, 0.05, -0.02, 0.01, 0.005]),
    "FULL_OPENCV": (
        CameraModelId.FULL_OPENCV,
        [810, 790, 500, 400, 0.01, -0.001, 0.0005, -0.0003, 0.0001, -0.00001, 0.000001, -0.0000001],
    ),
    "FOV": (CameraModelId.FOV, [810, 790, 500, 400, 0.9]),
    "SIMPLE_RADIAL_FISHEYE": (CameraModelId.SIMPLE_RADIAL_FISHEYE, [800, 500, 400, 0.01]),
    "RADIAL_FISHEYE": (CameraModelId.RADIAL_FISHEYE, [800, 500, 400, 0.01, -0.001]),
    "THIN_PRISM_FISHEYE": (
        CameraModelId.THIN_PRISM_FISHEYE,
        [810, 790, 500, 400, 0.05, -0.02, 0.002, -0.001, 0.01, 0.005, 0.002, -0.001],
    ),
}


POINTS_CAM = np.array(
    [
        [0.0, 0.0, 1.0],
        [0.1, 0.2, 1.0],
        [-0.25, 0.15, 1.3],
        [0.5, -0.35, 2.0],
        [1.0, 0.7, 1.2],
    ],
    dtype=np.float64,
)


def _make_cameras(model_id, params):
    cameras = Cameras(num_cameras=1)
    cameras.widths[0] = 1000
    cameras.heights[0] = 800
    cameras.set_params(0, np.array(params, dtype=np.float64), model_id)
    return cameras


def _params_without_principal_point(model_id, params):
    pp_indices = set(get_camera_model_info(model_id)["pp"])
    return [value for index, value in enumerate(params) if index not in pp_indices]


@pytest.mark.parametrize("model_name", MODEL_CASES)
def test_numpy_camera_projection_matches_pycolmap(model_name):
    model_id, params = MODEL_CASES[model_name]
    cameras = _make_cameras(model_id, params)
    reference = pycolmap.Camera(model=model_name, width=1000, height=800, params=params)

    projected = cameras.cam2img(POINTS_CAM, 0)
    expected = np.array([reference.img_from_cam(point) for point in POINTS_CAM])
    np.testing.assert_allclose(projected, expected, atol=1e-6, rtol=0)

    unprojected = cameras.img2cam(expected, 0)
    expected_unprojected = np.array([reference.cam_from_img(point) for point in expected])
    np.testing.assert_allclose(unprojected, expected_unprojected, atol=1e-6, rtol=0)


@pytest.mark.parametrize("model_name", MODEL_CASES)
def test_torch_reprojection_matches_pycolmap(model_name):
    model_id, params = MODEL_CASES[model_name]
    reference = pycolmap.Camera(model=model_name, width=1000, height=800, params=params)

    points = torch.tensor(POINTS_CAM, dtype=torch.float64)
    extrinsics = pp.identity_SE3(points.shape[0]).tensor().to(torch.float64)
    intrinsics = torch.tensor(
        [_params_without_principal_point(model_id, params)] * points.shape[0],
        dtype=torch.float64,
    )
    principal_point_np = np.tile(np.array(params)[get_camera_model_info(model_id)["pp"]], (points.shape[0], 1))
    principal_point = torch.tensor(principal_point_np, dtype=torch.float64)

    expected = np.array([reference.img_from_cam(point) for point in POINTS_CAM])

    projected = reproject_funcs_no_depth[model_id.value](points, extrinsics, intrinsics, principal_point)
    assert torch.isfinite(projected).all()
    np.testing.assert_allclose(projected.detach().numpy(), expected, atol=1e-6, rtol=0)

    projected_with_depth = reproject_funcs[model_id.value](points, extrinsics, intrinsics, principal_point)
    assert torch.isfinite(projected_with_depth).all()
    np.testing.assert_allclose(projected_with_depth.detach().numpy()[:, :2], expected, atol=1e-6, rtol=0)
    np.testing.assert_allclose(projected_with_depth.detach().numpy()[:, 2], POINTS_CAM[:, 2], atol=1e-12, rtol=0)
