from __future__ import annotations

import csv
import importlib.util
import os
import shutil
import subprocess
import sys
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pytest

from instantsfm.controllers.config import Config
from instantsfm.controllers.data_reader import ReadColmapDatabase
from instantsfm.processors.view_graph_calibration import (
    BuildViewGraphCalibrationInputs,
    EvaluateViewGraphCalibrationObjective,
    SolveViewGraphCalibration,
)
from instantsfm.processors.view_graph_manipulation import DecomposeRelPose, UpdateImagePairsConfig


ETH3D_DB_ROOT = ROOT / "db" / "eth3d" / "dslr"
ETH3D_DATASET_ROOT = ROOT / "datasets" / "eth3d" / "dslr"
STATUS_TSV = ROOT / "db" / "eth3d_dslr_feature_matching_status.tsv"
COLMAP_BIN = shutil.which("colmap")
REL_TOL = 2e-2
ABS_TOL = 1e-8


@lru_cache(maxsize=1)
def _load_eth3d_eval_module():
    spec = importlib.util.spec_from_file_location(
        "instantsfm_eth3d_eval",
        ROOT / "tools" / "eval_eth3d_pose_auc.py",
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


@lru_cache(maxsize=1)
def _discover_scenes() -> tuple[str, ...]:
    env_scenes = os.environ.get("INSTANTSFM_ETH3D_SCENES", "").strip()
    if env_scenes:
        return tuple(scene.strip() for scene in env_scenes.split(",") if scene.strip())

    scenes = []
    with STATUS_TSV.open() as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            if row["status"] in ("ok", "skipped"):
                scenes.append(row["scene"])
    return tuple(scenes)



def _prepare_scene_dir(tmp_path: Path, scene: str, scheme: str) -> Path:
    scene_dir = tmp_path / scheme / scene
    scene_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(ETH3D_DB_ROOT / scene / "database.db", scene_dir / "database.db")

    images_link = scene_dir / "images"
    if not images_link.exists():
        images_link.symlink_to(ETH3D_DATASET_ROOT / scene / "images")

    calib_link = scene_dir / "dslr_calibration_undistorted"
    if not calib_link.exists():
        calib_link.symlink_to(ETH3D_DATASET_ROOT / scene / "dslr_calibration_undistorted")

    if scheme == "gt_intrinsics":
        eth3d_eval = _load_eth3d_eval_module()
        _, ok, message = eth3d_eval.inject_gt_intrinsics_for_scene(scene_dir, force_reinject=False)
        assert ok, message

    return scene_dir



def _load_prepared_problem(scene_dir: Path):
    view_graph, cameras, images, feature_name, _ = ReadColmapDatabase(str(scene_dir / "database.db"))
    UpdateImagePairsConfig(view_graph, cameras, images)
    DecomposeRelPose(view_graph, cameras, images)
    config = Config(feature_name)
    calibration_inputs = BuildViewGraphCalibrationInputs(view_graph, cameras, images)
    return view_graph, cameras, images, config, calibration_inputs



def _run_colmap_view_graph_calibrator(database_path: Path, config: Config):
    assert COLMAP_BIN is not None
    options = config.VIEW_GRAPH_CALIBRATOR_OPTIONS
    cmd = [
        COLMAP_BIN,
        "view_graph_calibrator",
        "--database_path",
        str(database_path),
        "--cross_validate_prior_focal_lengths",
        "1",
        "--min_calibrated_pair_ratio",
        "0.5",
        "--reestimate_relative_pose",
        "0",
        "--min_focal_length_ratio",
        str(options["thres_lower_ratio"]),
        "--max_focal_length_ratio",
        str(options["thres_higher_ratio"]),
        "--max_calibration_error",
        str(options["thres_two_view_error"]),
        "--default_random_seed",
        "0",
    ]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, check=False)
    assert proc.returncode == 0, proc.stdout



def _evaluate_colmap_loss(database_path: Path, calibration_inputs, config: Config):
    _, cameras, _, _, _ = ReadColmapDatabase(str(database_path))
    return EvaluateViewGraphCalibrationObjective(
        calibration_inputs,
        cameras,
        config.VIEW_GRAPH_CALIBRATOR_OPTIONS["thres_loss_function"],
    )



def _assert_not_worse_than_colmap(tmp_path: Path, scheme: str):
    failures = []
    for scene in _discover_scenes():
        scene_dir = _prepare_scene_dir(tmp_path, scene, scheme)
        view_graph, cameras, images, config, calibration_inputs = _load_prepared_problem(scene_dir)

        instant_result = SolveViewGraphCalibration(
            view_graph,
            cameras,
            images,
            config.VIEW_GRAPH_CALIBRATOR_OPTIONS,
        )

        colmap_db_path = scene_dir / "database_colmap.db"
        shutil.copy2(scene_dir / "database.db", colmap_db_path)
        _run_colmap_view_graph_calibrator(colmap_db_path, config)
        colmap_raw_loss, colmap_robust_loss = _evaluate_colmap_loss(
            colmap_db_path,
            calibration_inputs,
            config,
        )

        tolerance = max(ABS_TOL, REL_TOL * max(abs(colmap_robust_loss), 1.0))
        if instant_result.robust_loss > colmap_robust_loss + tolerance:
            failures.append(
                (
                    scene,
                    instant_result.robust_loss,
                    colmap_robust_loss,
                    instant_result.raw_loss,
                    colmap_raw_loss,
                )
            )

    assert not failures, "\n".join(
        f"{scene}: instant robust={instant_robust:.12f}, colmap robust={colmap_robust:.12f}, "
        f"instant raw={instant_raw:.12f}, colmap raw={colmap_raw:.12f}"
        for scene, instant_robust, colmap_robust, instant_raw, colmap_raw in failures
    )


@pytest.mark.skipif(COLMAP_BIN is None, reason="colmap executable is required for ETH3D view-graph calibration regression tests")
def test_view_graph_calibration_eth3d_no_intrinsics_not_worse_than_colmap(tmp_path):
    _assert_not_worse_than_colmap(tmp_path, "no_intrinsics")


@pytest.mark.skipif(COLMAP_BIN is None, reason="colmap executable is required for ETH3D view-graph calibration regression tests")
def test_view_graph_calibration_eth3d_gt_intrinsics_not_worse_than_colmap(tmp_path):
    _assert_not_worse_than_colmap(tmp_path, "gt_intrinsics")


@pytest.mark.skipif(COLMAP_BIN is None, reason="colmap executable is required for ETH3D view-graph calibration regression tests")
def test_view_graph_calibration_gt_intrinsics_keeps_prior_focals_fixed(tmp_path):
    drifted_scenes = []
    for scene in _discover_scenes():
        scene_dir = _prepare_scene_dir(tmp_path, scene, "gt_intrinsics")
        view_graph, cameras, images, config, _ = _load_prepared_problem(scene_dir)
        initial_focals = [float(cam.focal()) for cam in cameras]
        result = SolveViewGraphCalibration(
            view_graph,
            cameras,
            images,
            config.VIEW_GRAPH_CALIBRATOR_OPTIONS,
        )
        if not all(abs(a - b) <= ABS_TOL for a, b in zip(initial_focals, result.optimized_focals)):
            drifted_scenes.append((scene, initial_focals, result.optimized_focals.tolist()))

    assert not drifted_scenes, "\n".join(
        f"{scene}: initial={initial}, optimized={optimized}"
        for scene, initial, optimized in drifted_scenes
    )
