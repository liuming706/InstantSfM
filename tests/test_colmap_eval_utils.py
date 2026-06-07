from __future__ import annotations

from collections import OrderedDict

import numpy as np

from instantsfm.eval.colmap_eval.evaluation.utils import (
    compute_abs_errors,
    compute_auc,
    compute_rel_errors,
)

if not hasattr(np, "acos"):
    np.acos = np.arccos  # type: ignore[attr-defined]


class _FakeRotation:
    def __init__(self, angle_rad: float):
        self._angle_rad = angle_rad

    def angle(self) -> float:
        return self._angle_rad


class _FakeTransform:
    def __init__(self, translation, angle_rad: float = 0.0):
        self.translation = np.array(translation, dtype=np.float64)
        self._angle_rad = angle_rad
        self.rotation = _FakeRotation(angle_rad)

    def inverse(self) -> "_FakeTransform":
        return _FakeTransform(-self.translation, -self._angle_rad)

    def __mul__(self, other: "_FakeTransform") -> "_FakeTransform":
        return _FakeTransform(
            self.translation + other.translation,
            self._angle_rad + other._angle_rad,
        )


class _MethodImage:
    def __init__(self, image_id: int, name: str, transform: _FakeTransform):
        self.image_id = image_id
        self.name = name
        self._transform = transform

    def cam_from_world(self) -> _FakeTransform:
        return self._transform


class _PropertyImage:
    def __init__(self, image_id: int, name: str, transform: _FakeTransform):
        self.image_id = image_id
        self.name = name
        self.cam_from_world = transform


class _FakeReconstruction:
    def __init__(self, images):
        self.images = OrderedDict((image.image_id, image) for image in images)

    def num_images(self) -> int:
        return len(self.images)


def test_compute_abs_errors_accepts_method_cam_from_world():
    transform = _FakeTransform([1.0, 2.0, 3.0], angle_rad=0.25)
    sparse_gt = _FakeReconstruction([_MethodImage(1, "im1.jpg", transform)])
    sparse = _FakeReconstruction([_MethodImage(1, "im1.jpg", transform)])

    dts, dRs = compute_abs_errors(sparse_gt, sparse)

    assert np.array_equal(dts, np.array([0.0]))
    assert np.array_equal(dRs, np.array([0.0]))


def test_compute_rel_errors_accepts_method_and_property_cam_from_world():
    gt_images = [
        _MethodImage(1, "gt/im1.jpg", _FakeTransform([0.0, 0.0, 0.0])),
        _MethodImage(2, "gt/im2.jpg", _FakeTransform([1.0, 0.0, 0.0])),
    ]
    sparse_images = [
        _PropertyImage(1, "pred/im1.jpg", _FakeTransform([0.0, 0.0, 0.0])),
        _PropertyImage(2, "pred/im2.jpg", _FakeTransform([1.0, 0.0, 0.0])),
    ]
    sparse_gt = _FakeReconstruction(gt_images)
    sparse = _FakeReconstruction(sparse_images)

    dts, dRs = compute_rel_errors(
        sparse_gt,
        sparse,
        min_proj_center_dist=1e-6,
    )

    assert np.array_equal(dts, np.array([0.0, 0.0]))
    assert np.array_equal(dRs, np.array([0.0, 0.0]))


def test_compute_auc_works_without_numpy_trapezoid_alias():
    aucs = compute_auc(np.array([1.0, 2.0]), [3.0], min_error=0.0)

    assert np.allclose(aucs, np.array([60.60606061]))
