from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import pyceres
import torch
import tqdm
from scipy.optimize import minimize_scalar

from instantsfm.scene.defs import ConfigurationType, ViewGraph
from instantsfm.utils.cost_function import (
    FetzerFocalLengthCostFunction,
    FetzerFocalLengthSameCameraCostFunction,
    fetzer_cost,
    fetzer_ds,
)
from instantsfm.utils.optimization_models import FetzerCalibrationModel

# used by torch LM
import pypose as pp
from pypose.optim.kernel import Cauchy
from bae.utils.pysolvers import PCG
from bae.optim import LM


DEFAULT_MIN_CALIBRATED_PAIR_RATIO = 0.5


@dataclass(frozen=True)
class ViewGraphCalibrationInput:
    pair_id: Tuple[int, int]
    camera_id1: int
    camera_id2: int
    F: np.ndarray
    d01: np.ndarray
    d12: np.ndarray


@dataclass
class ViewGraphCalibrationResult:
    success: bool
    initial_focals: np.ndarray
    optimized_focals: np.ndarray
    calibration_errors_sq: Dict[Tuple[int, int], float]
    num_rejected_cameras: int
    num_invalid_pairs: int
    raw_loss: float
    robust_loss: float


def _fundamental_from_essential(E: np.ndarray, camera1, camera2) -> np.ndarray:
    K1_inv = np.linalg.inv(camera1.get_K())
    K2_inv_t = np.linalg.inv(camera2.get_K()).T
    return K2_inv_t @ E @ K1_inv



def _essential_from_fundamental(F: np.ndarray, camera1, camera2) -> np.ndarray:
    return camera2.get_K().T @ F @ camera1.get_K()



def _compute_pair_residuals(
    calibration_input: ViewGraphCalibrationInput,
    focal_length1: float,
    focal_length2: float,
):
    fi_sq = focal_length1 * focal_length1
    fj_sq = focal_length2 * focal_length2

    di = fj_sq * calibration_input.d01[0] + calibration_input.d01[1]
    dj = fi_sq * calibration_input.d12[0] + calibration_input.d12[2]
    if di == 0:
        di = 1e-6
    if dj == 0:
        dj = 1e-6

    k0_01 = -(fj_sq * calibration_input.d01[2] + calibration_input.d01[3]) / di
    k1_12 = -(fi_sq * calibration_input.d12[1] + calibration_input.d12[3]) / dj

    residual0 = (fi_sq - k0_01) / fi_sq
    residual1 = (fj_sq - k1_12) / fj_sq
    return residual0, residual1


def _compute_pair_error_sq(
    calibration_input: ViewGraphCalibrationInput,
    focal_length1: float,
    focal_length2: float,
) -> float:
    residual0, residual1 = _compute_pair_residuals(
        calibration_input,
        focal_length1,
        focal_length2,
    )
    return residual0 * residual0 + residual1 * residual1



def BuildViewGraphCalibrationInputs(view_graph: ViewGraph, cameras, images):
    inputs = []
    for pair_id, image_pair in view_graph.image_pairs.items():
        if not image_pair.is_valid:
            continue
        if image_pair.config not in (ConfigurationType.CALIBRATED, ConfigurationType.UNCALIBRATED):
            continue

        image1 = images[image_pair.image_id1]
        image2 = images[image_pair.image_id2]
        camera1 = cameras[image1.cam_id]
        camera2 = cameras[image2.cam_id]

        F = image_pair.F
        if image_pair.config == ConfigurationType.CALIBRATED and image_pair.E is not None and np.any(image_pair.E):
            F = _fundamental_from_essential(image_pair.E, camera1, camera2)
            image_pair.F = F

        K0 = np.array(
            [[1.0, 0.0, camera1.principal_point[0]], [0.0, 1.0, camera1.principal_point[1]], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
        K1 = np.array(
            [[1.0, 0.0, camera2.principal_point[0]], [0.0, 1.0, camera2.principal_point[1]], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
        ds = fetzer_ds(K1.T @ F @ K0)
        inputs.append(
            ViewGraphCalibrationInput(
                pair_id=pair_id,
                camera_id1=image1.cam_id,
                camera_id2=image2.cam_id,
                F=F,
                d01=np.asarray(ds[0], dtype=np.float64),
                d12=np.asarray(ds[2], dtype=np.float64),
            )
        )

    return inputs



def EvaluateViewGraphCalibrationObjective(calibration_inputs, cameras, loss_scale: float):
    raw_loss = 0.0
    robust_loss = 0.0
    loss_scale_sq = loss_scale * loss_scale

    for calibration_input in calibration_inputs:
        focal_length1 = float(np.mean(cameras[calibration_input.camera_id1].focal_length))
        focal_length2 = float(np.mean(cameras[calibration_input.camera_id2].focal_length))
        error_sq = _compute_pair_error_sq(calibration_input, focal_length1, focal_length2)
        raw_loss += error_sq
        robust_loss += 0.5 * loss_scale_sq * np.log1p(error_sq / loss_scale_sq)

    return raw_loss, robust_loss



def SolveViewGraphCalibration(view_graph: ViewGraph, cameras, images, VIEW_GRAPH_CALIBRATOR_OPTIONS):
    calibration_inputs = BuildViewGraphCalibrationInputs(view_graph, cameras, images)
    initial_focals = np.array([np.mean(cam.focal_length) for cam in cameras], dtype=np.float64)
    optimized_focals = initial_focals.copy()

    if len(calibration_inputs) == 0:
        print('No valid image pairs for view graph calibration; skipping.')
        raw_loss, robust_loss = EvaluateViewGraphCalibrationObjective(
            calibration_inputs,
            cameras,
            VIEW_GRAPH_CALIBRATOR_OPTIONS['thres_loss_function'],
        )
        return ViewGraphCalibrationResult(
            success=True,
            initial_focals=initial_focals,
            optimized_focals=optimized_focals,
            calibration_errors_sq={},
            num_rejected_cameras=0,
            num_invalid_pairs=0,
            raw_loss=raw_loss,
            robust_loss=robust_loss,
        )

    problem = pyceres.Problem()
    options = pyceres.SolverOptions()
    loss_function = pyceres.CauchyLoss(VIEW_GRAPH_CALIBRATOR_OPTIONS['thres_loss_function'])
    if len(cameras) < 50:
        options.linear_solver_type = pyceres.LinearSolverType.DENSE_NORMAL_CHOLESKY
    else:
        options.linear_solver_type = pyceres.LinearSolverType.SPARSE_NORMAL_CHOLESKY

    parameter_blocks = {}
    for calibration_input in calibration_inputs:
        cam_id1 = calibration_input.camera_id1
        cam_id2 = calibration_input.camera_id2
        parameter_blocks.setdefault(cam_id1, optimized_focals[cam_id1:cam_id1 + 1])
        parameter_blocks.setdefault(cam_id2, optimized_focals[cam_id2:cam_id2 + 1])

        if cam_id1 == cam_id2:
            cost_function = FetzerFocalLengthSameCameraCostFunction(
                calibration_input.F,
                cameras[cam_id1].principal_point,
            )
            problem.add_residual_block(cost_function, loss_function, [parameter_blocks[cam_id1]])
        else:
            cost_function = FetzerFocalLengthCostFunction(
                calibration_input.F,
                cameras[cam_id1].principal_point,
                cameras[cam_id2].principal_point,
            )
            problem.add_residual_block(
                cost_function,
                loss_function,
                [parameter_blocks[cam_id1], parameter_blocks[cam_id2]],
            )

    num_variable_cameras = 0
    for cam_id, parameter_block in parameter_blocks.items():
        problem.set_parameter_lower_bound(parameter_block, 0, 1e-3)
        if cameras[cam_id].has_prior_focal_length:
            problem.set_parameter_block_constant(parameter_block)
        else:
            num_variable_cameras += 1

    options.max_num_iterations = VIEW_GRAPH_CALIBRATOR_OPTIONS['max_num_iterations']
    options.function_tolerance = VIEW_GRAPH_CALIBRATOR_OPTIONS['function_tolerance']

    success = True
    if num_variable_cameras == 0:
        print('All connected cameras have prior focal length, skipping view graph calibration')
    else:
        variable_camera_ids = sorted(
            cam_id for cam_id in parameter_blocks if not cameras[cam_id].has_prior_focal_length
        )
        if len(variable_camera_ids) == 1:
            variable_cam_id = variable_camera_ids[0]
            initial_focal = initial_focals[variable_cam_id]
            loss_scale_sq = VIEW_GRAPH_CALIBRATOR_OPTIONS['thres_loss_function'] ** 2

            def scalar_objective(focal):
                total = 0.0
                for calibration_input in calibration_inputs:
                    focal1 = initial_focals[calibration_input.camera_id1]
                    focal2 = initial_focals[calibration_input.camera_id2]
                    if calibration_input.camera_id1 == variable_cam_id:
                        focal1 = focal
                    if calibration_input.camera_id2 == variable_cam_id:
                        focal2 = focal
                    error_sq = _compute_pair_error_sq(
                        calibration_input,
                        float(focal1),
                        float(focal2),
                    )
                    total += 0.5 * loss_scale_sq * np.log1p(error_sq / loss_scale_sq)
                return total

            upper_bound = max(
                1.0,
                initial_focal * VIEW_GRAPH_CALIBRATOR_OPTIONS['thres_higher_ratio'],
            )
            scipy_result = minimize_scalar(
                scalar_objective,
                bounds=(1e-3, upper_bound),
                method='bounded',
                options={'xatol': VIEW_GRAPH_CALIBRATOR_OPTIONS['function_tolerance'], 'maxiter': VIEW_GRAPH_CALIBRATOR_OPTIONS['max_num_iterations']},
            )
            optimized_focals[variable_cam_id] = float(scipy_result.x)
            print(
                'SciPy minimize_scalar report:',
                f'nfev={scipy_result.nfev}',
                f'initial_cost={scalar_objective(initial_focal):.6e}',
                f'final_cost={float(scipy_result.fun):.6e}',
                f'success={bool(scipy_result.success)}',
            )
            success = bool(scipy_result.success)
            if not success:
                print('View graph calibration scalar solve did not converge cleanly')
        else:
            summary = pyceres.SolverSummary()
            pyceres.solve(options, problem, summary)
            print(summary.BriefReport())
            success = getattr(summary, 'is_solution_usable', lambda: True)()
            if not success:
                print('View graph calibration solver failed to produce a usable solution')

    counter = 0
    for cam_id, parameter_block in parameter_blocks.items():
        focal = float(parameter_block[0])
        focal_ratio = focal / initial_focals[cam_id]
        if (
            focal_ratio < VIEW_GRAPH_CALIBRATOR_OPTIONS['thres_lower_ratio']
            or focal_ratio > VIEW_GRAPH_CALIBRATOR_OPTIONS['thres_higher_ratio']
        ):
            optimized_focals[cam_id] = initial_focals[cam_id]
            counter += 1
        else:
            optimized_focals[cam_id] = focal

    for idx, cam in enumerate(cameras):
        focal = float(optimized_focals[idx])
        cam.focal_length = np.array([focal, focal], dtype=np.float64)
        cam.has_prior_focal_length = True

    print(f'{counter} cameras are rejected in view graph calibration')

    calibration_errors_sq = {}
    invalid_counter = 0
    thres_two_view_error_sq = VIEW_GRAPH_CALIBRATOR_OPTIONS['thres_two_view_error'] ** 2

    for calibration_input in calibration_inputs:
        error_sq = _compute_pair_error_sq(
            calibration_input,
            optimized_focals[calibration_input.camera_id1],
            optimized_focals[calibration_input.camera_id2],
        )
        calibration_errors_sq[calibration_input.pair_id] = error_sq

        image_pair = view_graph.image_pairs[calibration_input.pair_id]
        image_pair.F = calibration_input.F
        if error_sq > thres_two_view_error_sq:
            invalid_counter += 1
            image_pair.is_valid = False
            image_pair.config = ConfigurationType.DEGENERATE
        else:
            image_pair.is_valid = True
            image_pair.config = ConfigurationType.CALIBRATED
            image1 = images[image_pair.image_id1]
            image2 = images[image_pair.image_id2]
            image_pair.E = _essential_from_fundamental(
                image_pair.F,
                cameras[image1.cam_id],
                cameras[image2.cam_id],
            )

    print(f'invalid / total number of two view geometry: {invalid_counter} / {len(calibration_inputs)}')

    raw_loss, robust_loss = EvaluateViewGraphCalibrationObjective(
        calibration_inputs,
        cameras,
        VIEW_GRAPH_CALIBRATOR_OPTIONS['thres_loss_function'],
    )
    return ViewGraphCalibrationResult(
        success=success,
        initial_focals=initial_focals,
        optimized_focals=optimized_focals.copy(),
        calibration_errors_sq=calibration_errors_sq,
        num_rejected_cameras=counter,
        num_invalid_pairs=invalid_counter,
        raw_loss=raw_loss,
        robust_loss=robust_loss,
    )


class TorchVGC():
    def __init__(self, device='cuda:0'):
        self.device = device

    def Optimize(self, view_graph:ViewGraph, cameras, images, VIEW_GRAPH_CALIBRATOR_OPTIONS):
        cost_fn = fetzer_cost

        valid_image_pairs = {pair_id: image_pair for pair_id, image_pair in view_graph.image_pairs.items()
                             if image_pair.is_valid and image_pair.config in [ConfigurationType.CALIBRATED, ConfigurationType.UNCALIBRATED]}
        focals = torch.tensor(np.array([np.mean(cam.focal_length) for cam in cameras]), dtype=torch.float64).to(self.device).unsqueeze(-1)
        self.camera_has_prior = torch.tensor([cam.has_prior_focal_length for cam in cameras], dtype=torch.bool).to(self.device)
        # TODO: Only support all cameras have prior focal length. If some cameras have prior focal length while others do not, 
        # they will be optimized together, which is not a good idea.
        if torch.all(self.camera_has_prior):
            print('All cameras have prior focal length, skipping view graph calibration')
            return

        ds_list = []
        camera_indices1_list = []
        camera_indices2_list = []
        for image_pair in valid_image_pairs.values():
            # add both directions
            image1, image2 = images[image_pair.image_id1], images[image_pair.image_id2]
            cam_id1, cam_id2 = image1.cam_id, image2.cam_id
            cam1, cam2 = cameras[cam_id1], cameras[cam_id2]
            principal_point0, principal_point1 = cam1.principal_point, cam2.principal_point
            K0 = np.array([[1, 0, principal_point0[0]], [0, 1, principal_point0[1]], [0, 0, 1]])
            K1 = np.array([[1, 0, principal_point1[0]], [0, 1, principal_point1[1]], [0, 0, 1]])
            i1_G_i0 = K1.T @ image_pair.F @ K0
            ds = fetzer_ds(i1_G_i0)
            ds_list.append(ds)
            camera_indices1_list.append(cam_id1)
            camera_indices2_list.append(cam_id2)
            i0_G_i1 = i1_G_i0.T
            ds = fetzer_ds(i0_G_i1)
            ds_list.append(ds)
            camera_indices1_list.append(cam_id2)
            camera_indices2_list.append(cam_id1)
        
        ds = torch.tensor(np.array(ds_list), dtype=torch.float64).to(self.device).unsqueeze(1)
        camera_indices1 = torch.tensor(np.array(camera_indices1_list), dtype=torch.int64).to(self.device).flatten()
        camera_indices2 = torch.tensor(np.array(camera_indices2_list), dtype=torch.int64).to(self.device).flatten()

        model = FetzerCalibrationModel(focals, cost_fn)
        strategy = pp.optim.strategy.TrustRegion(radius=1e2, max=1e6, up=2.0, down=0.5**4)
        sparse_solver = PCG(tol=1e-5) # cuSolverSP()
        cauchy_kernel = Cauchy(VIEW_GRAPH_CALIBRATOR_OPTIONS['thres_loss_function'])
        optimizer = LM(model, strategy=strategy, solver=sparse_solver, kernel=cauchy_kernel, reject=30)

        input = {
            "ds": ds,
            "camera_indices1": camera_indices1,
            "camera_indices2": camera_indices2
        }

        window_size = 3
        loss_history = []
        progress_bar = tqdm.trange(VIEW_GRAPH_CALIBRATOR_OPTIONS['max_num_iterations'])
        for _ in progress_bar:
            loss = optimizer.step(input)
            torch.set_printoptions(threshold=torch.inf)
            print(f'focals: {model.focals}, loss: {loss.item()}')
            loss_history.append(loss.item())
            if len(loss_history) >= 2*window_size:
                avg_recent = np.mean(loss_history[-window_size:])
                avg_previous = np.mean(loss_history[-2*window_size:-window_size])
                improvement = (avg_previous - avg_recent) / avg_previous
                if abs(improvement) < VIEW_GRAPH_CALIBRATOR_OPTIONS['function_tolerance']:
                    break
            progress_bar.set_postfix({"loss": loss.item()})
        progress_bar.close()

        focals_ = focals.detach().cpu().numpy().squeeze()
        counter = 0
        for cam, focal in zip(cameras, focals_):
            if (focal / np.mean(cam.focal_length) < VIEW_GRAPH_CALIBRATOR_OPTIONS['thres_lower_ratio'] or 
                focal / np.mean(cam.focal_length) > VIEW_GRAPH_CALIBRATOR_OPTIONS['thres_higher_ratio']):
                counter += 1
                continue
            cam.has_refined_focal_length = True
            cam.focal_length = np.array([focal, focal])
        
        print(f'{counter} cameras are rejected in view graph calibration')

        thres_two_view_error_sq = VIEW_GRAPH_CALIBRATOR_OPTIONS['thres_two_view_error'] ** 2

        # manually calculate the residuals
        loss = model.forward(ds, camera_indices1, camera_indices2).detach().cpu().numpy()
        invalid_counter = 0
        loss_sq = np.sum(loss ** 2, axis=-1)
        for idx, (pair_id, image_pair) in enumerate(valid_image_pairs.items()):
            if loss_sq[idx*2] > thres_two_view_error_sq:
                invalid_counter += 1
                view_graph.image_pairs[pair_id].is_valid = False
        print(f'invalid / total number of two view geometry: {invalid_counter} / {len(valid_image_pairs)}')
