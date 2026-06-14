import os
import subprocess
import time
import csv
import shutil
import argparse
import numpy as np
from scipy.spatial import KDTree

from instantsfm.utils.read_write_model import (
    read_images_binary,
    read_images_text,
    read_cameras_text,
    read_cameras_binary,
    read_points3D_binary,
    read_points3D_text,
)

try:
    import pycolmap
    HAS_PYCOLMAP = True
except ImportError:
    HAS_PYCOLMAP = False
    print("Warning: pycolmap not available. Pose evaluation metrics will be disabled.")

# Dataset registry
DATASET_CONFIGS = {
    "360": {"path": "dataset/360", "multi_camera_rig": False},
    "ScanNet": {"path": "dataset/ScanNet", "multi_camera_rig": False},
    "DTU": {"path": "dataset/dtu/dtu_testing", "multi_camera_rig": False},
    "ETH3D": {"path": "dataset/eth3d/dslr", "multi_camera_rig": False},
    "IMC": {"path": "dataset/IMC", "multi_camera_rig": False},
    "TanksAndTemples": {"path": "dataset/tt/Intermediate", "multi_camera_rig": False},
    "KITTI": {"path": "dataset/KITTI", "multi_camera_rig": False},
    "waymo": {"path": "dataset/waymo", "multi_camera_rig": True},
}

# ---------------- dataset helpers ----------------
def list_scenes_for_dataset(dataset_key, dataset_path):
    if dataset_key == "360":
        return ["bicycle", "bonsai", "counter", "garden", "kitchen", "room", "stump"]
    if os.path.exists(dataset_path):
        entries = [d for d in os.listdir(dataset_path)
                   if os.path.isdir(os.path.join(dataset_path, d)) and not d.startswith('.')]
        return sorted(entries)
    return []

def get_scene_paths(dataset_key, dataset_path, scene, is_multi_camera_rig=False):
    scene_root = os.path.join(dataset_path, scene)
    
    # For multi-camera rig datasets, use scene_root directly as img_path
    if is_multi_camera_rig:
        img_path = scene_root
    else:
        # For regular datasets, append images or color folder
        img_path = os.path.join(scene_root, "images")
        if not os.path.exists(img_path):
            img_path = os.path.join(scene_root, "color")
    
    db_path = os.path.join(scene_root, "database.db")
    return scene_root, img_path, db_path

def ensure_database(scene_root, img_path, db_path, force_recompute=False, is_multi_camera_rig=False):
    if os.path.exists(db_path) and not force_recompute:
        print(f"Database exists: {db_path} (skip extraction)")
        return True

    print(f"Generating COLMAP database for scene {scene_root} (images: {img_path})")
    # feature extractor expects --image_path and --database_path
    # For multi-camera rigs, use single_camera_per_folder=1 to handle separate folders
    single_camera_value = "1" if is_multi_camera_rig else "0"
    run_cmd(["colmap", "feature_extractor", "--database_path", db_path, "--image_path", img_path, "--ImageReader.single_camera_per_folder", single_camera_value])
    run_cmd(["colmap", "exhaustive_matcher", "--database_path", db_path])
    return os.path.exists(db_path)

# ---------------- utility helpers ----------------
def run_cmd(cmd, cwd=None, check=True):
    return subprocess.run(cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=check)


def chamfer_distance_kdtree(point_cloud_a, point_cloud_b):
    point_cloud_a = np.asarray(point_cloud_a)
    point_cloud_b = np.asarray(point_cloud_b)

    tree_a = KDTree(point_cloud_a)
    tree_b = KDTree(point_cloud_b)

    dist_a_to_b, _ = tree_b.query(point_cloud_a)
    dist_b_to_a, _ = tree_a.query(point_cloud_b)

    return np.mean(dist_a_to_b) + np.mean(dist_b_to_a)


def load_point_cloud(file_path):
    if file_path.endswith(".ply"):
        import open3d as o3d

        point_cloud = o3d.io.read_point_cloud(file_path)
        return np.asarray(point_cloud.points)
    if file_path.endswith("points3D.txt"):
        points3d = read_points3D_text(file_path)
        return np.stack([point.xyz for point in points3d.values()])
    if file_path.endswith("points3D.bin"):
        points3d = read_points3D_binary(file_path)
        return np.stack([point.xyz for point in points3d.values()])
    return None

def safe_rmtree(path):
    if os.path.exists(path):
        shutil.rmtree(path)

def extract_poses_from_images_bin(model_path):
    images_bin = os.path.join(model_path, "images.bin")
    if not os.path.exists(images_bin):
        return None
    images = read_images_binary(images_bin)
    txt_file = os.path.join(model_path, "pose_merged.txt")
    with open(txt_file, 'w') as f:
        for image_id in images:
            im = images[image_id]
            t = im.tvec if hasattr(im, "tvec") else im.t
            f.write(f"{im.name} {t[0]} {t[1]} {t[2]}\n")
    return txt_file

def extract_poses_from_images_txt(model_path):
    """Extract poses from images.txt and create pose_merged.txt for alignment."""
    images_txt = os.path.join(model_path, "images.txt")
    if not os.path.exists(images_txt):
        return None
    images = read_images_text(images_txt)
    txt_file = os.path.join(model_path, "pose_merged.txt")
    with open(txt_file, 'w') as f:
        for image_id in images:
            im = images[image_id]
            t = im.tvec if hasattr(im, "tvec") else im.t
            f.write(f"{im.name} {t[0]} {t[1]} {t[2]}\n")
    return txt_file

def align_model(input_model, ref_pose_txt, alignment_max_error=4.0):
    aligned = f"{input_model}_aligned"
    safe_rmtree(aligned)
    os.makedirs(aligned, exist_ok=True)
    cmd = [
        "colmap", "model_aligner",
        "--input_path", input_model,
        "--output_path", aligned,
        "--ref_is_gps", "0",
        "--ref_images_path", ref_pose_txt,
        "--alignment_max_error", str(alignment_max_error)
    ]
    run_cmd(cmd)
    return aligned

def load_images_from_model(model_path):
    """Load images from a model directory, supporting both complete and incomplete GT.
    
    Returns:
        dict: image_name -> Image object with qvec, tvec, camera_id
        None: if loading failed
    """
    # Try binary format first
    images_bin = os.path.join(model_path, "images.bin")
    if os.path.exists(images_bin):
        return read_images_binary(images_bin)
    
    # Try text format (for incomplete GT like Waymo)
    images_txt = os.path.join(model_path, "images.txt")
    if os.path.exists(images_txt):
        return read_images_text(images_txt)
    
    # Try using pycolmap for complete reconstruction
    if HAS_PYCOLMAP:
        try:
            recon = pycolmap.Reconstruction(model_path)
            # Convert pycolmap images to dict format
            images_dict = {}
            for img_id, img in recon.images.items():
                # Create Image namedtuple compatible object
                from instantsfm.utils.read_write_model import Image as ImageTuple
                images_dict[img_id] = ImageTuple(
                    id=img.image_id,
                    qvec=img.cam_from_world.rotation.quat[[3,0,1,2]],  # pycolmap uses wxyz, convert to qwxyz
                    tvec=img.cam_from_world.translation,
                    camera_id=img.camera_id,
                    name=img.name,
                    xys=np.array([]),
                    point3D_ids=np.array([])
                )
            return images_dict
        except:
            return None
    
    return None

def find_points3d_in_model(model_path):
    candidates = [
        os.path.join(model_path, 'points3D.bin'),
        os.path.join(model_path, 'points3D.ply'),
        os.path.join(model_path, 'points3D.txt'),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return None

# ---------------- runners (methods) ----------------
def runner_colmap(dataset_path, dataset_key, options):
    start = time.time()
    out_dir = os.path.join(dataset_path, "sparse_colmap")
    safe_rmtree(out_dir)
    os.makedirs(out_dir, exist_ok=True)
    db = os.path.join(dataset_path, "database.db")
    
    # Handle multi-camera rig vs regular datasets
    is_multi_camera_rig = options.get("is_multi_camera_rig", False)
    if is_multi_camera_rig:
        img_path = dataset_path
    else:
        img_path = os.path.join(dataset_path, "images") if os.path.exists(os.path.join(dataset_path, "images")) else os.path.join(dataset_path, "color")
    
    run_cmd(["colmap", "mapper", "--database_path", db, "--image_path", img_path, "--output_path", out_dir])
    elapsed = time.time() - start
    model0 = os.path.join(out_dir, "0")
    return {"model_path": model0 if os.path.exists(model0) else out_dir, "time": elapsed}

def runner_glomap(dataset_path, dataset_key, options):
    start = time.time()
    out_dir = os.path.join(dataset_path, "sparse_glomap")
    safe_rmtree(out_dir)
    os.makedirs(out_dir, exist_ok=True)
    db = os.path.join(dataset_path, "database.db")
    
    # Handle multi-camera rig vs regular datasets
    is_multi_camera_rig = options.get("is_multi_camera_rig", False)
    if is_multi_camera_rig:
        img_path = dataset_path
    else:
        img_path = os.path.join(dataset_path, "images") if os.path.exists(os.path.join(dataset_path, "images")) else os.path.join(dataset_path, "color")
    
    run_cmd(["glomap", "mapper", "--database_path", db, "--image_path", img_path, "--output_path", out_dir], check=True)
    elapsed = time.time() - start
    model0 = os.path.join(out_dir, "0")
    return {"model_path": model0 if os.path.exists(model0) else out_dir, "time": elapsed}

def runner_instantsfm(dataset_path, dataset_key, options):
    start = time.time()
    out_dir = os.path.join(dataset_path, "sparse")
    safe_rmtree(out_dir)
    os.makedirs(out_dir, exist_ok=True)
    cmd = ["ins-sfm", "--data", dataset_path]
    run_cmd(cmd, check=True)
    elapsed = time.time() - start
    model0 = os.path.join(out_dir, "0")
    return {"model_path": model0 if os.path.exists(model0) else out_dir, "time": elapsed}

RUNNERS = {
    "colmap": runner_colmap,
    "glomap": runner_glomap,
    "instantsfm": runner_instantsfm,
}

# ---------------- metrics ----------------
def metric_timing(runner_result, dataset_path=None, **kwargs):
    return runner_result.get("time", None)

def metric_points_count(runner_result, dataset_path=None, **kwargs):
    model = runner_result.get("model_path")
    if not model:
        return None
    pts = find_points3d_in_model(model)
    if not pts:
        # try aligned
        aligned = model + "_aligned"
        pts = find_points3d_in_model(aligned)
        if not pts:
            return 0
    pc = load_point_cloud(pts)
    return len(pc) if pc is not None else 0

def metric_chamfer(runner_result, dataset_path, **kwargs):
    model = runner_result.get("model_path")
    if not model:
        return float("inf")
    # find GT pointcloud under dataset conventions (for 360: dataset/.../sparse_gt/0)
    gt_candidates = [
        os.path.join(dataset_path, "sparse_gt", "0", "points3D.bin"),
    ]
    gt_path = next((p for p in gt_candidates if os.path.exists(p)), None)
    # if gt poses exist, extract for alignment
    ref_pose_txt = None
    sparse_gt_images = os.path.join(dataset_path, "sparse_gt", "0")
    if os.path.exists(os.path.join(sparse_gt_images, "images.bin")):
        ref_pose_txt = extract_poses_from_images_bin(sparse_gt_images)
    if ref_pose_txt is None and os.path.exists(os.path.join(dataset_path, "pose_merged.txt")):
        ref_pose_txt = os.path.join(dataset_path, "pose_merged.txt")

    aligned = align_model(model, ref_pose_txt) if ref_pose_txt else model
    pts = find_points3d_in_model(aligned)
    if pts is None:
        return float("inf")
    if gt_path is None:
        return float("inf")
    gt_pc = load_point_cloud(gt_path)
    recon_pc = load_point_cloud(pts)
    if gt_pc is None or recon_pc is None:
        return float("inf")
    return chamfer_distance_kdtree(gt_pc, recon_pc)

def qvec2rotmat(qvec):
    """Convert quaternion to rotation matrix."""
    return np.array([
        [1 - 2 * qvec[2]**2 - 2 * qvec[3]**2,
         2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
         2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2]],
        [2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
         1 - 2 * qvec[1]**2 - 2 * qvec[3]**2,
         2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1]],
        [2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
         2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
         1 - 2 * qvec[1]**2 - 2 * qvec[2]**2]])

def compute_relative_pose(img1, img2):
    """Compute relative pose from img1 to img2.
    
    Returns:
        R_rel: Relative rotation matrix
        t_rel: Relative translation vector
    """
    R1 = qvec2rotmat(img1.qvec)
    R2 = qvec2rotmat(img2.qvec)
    t1 = img1.tvec
    t2 = img2.tvec
    
    # Compute relative pose: camera2_from_camera1
    # In COLMAP convention: x_cam = R @ X_world + t
    # Relative pose: t_rel = t2 - R_rel @ t1
    R_rel = R2 @ R1.T
    t_rel = t2 - R_rel @ t1
    
    return R_rel, t_rel

def rotation_matrix_angle(R):
    """Compute rotation angle from rotation matrix (in degrees)."""
    trace = np.trace(R)
    angle_rad = np.arccos(np.clip((trace - 1) / 2, -1, 1))
    return np.rad2deg(angle_rad)

def compute_rel_pose_errors(images_gt, images, min_proj_center_dist=0.1):
    """Compute relative pose errors (angular distance) between image pairs.
    
    NOTE: This function expects the input images to be already aligned to GT when calculating translation errors.
    The alignment should be done before calling this function.
    
    Args:
        images_gt: dict of GT images (image_id -> Image)
        images: dict of reconstructed images (image_id -> Image)
    """
    if images is None or len(images) == 0:
        return np.array([]), np.array([])
    
    # Create name-based lookup for reconstructed images
    images_by_name = {img.name.split('/')[-1]: img for img in images.values()}
    
    dts, dRs = [], []
    images_gt_list = list(images_gt.values())
    
    for this_img_gt in images_gt_list:
        this_name = this_img_gt.name.split('/')[-1]
        if this_name not in images_by_name:
            for _ in range(len(images_gt) - 1):
                dts.append(np.inf)
                dRs.append(180)
            continue
        
        this_img = images_by_name[this_name]
        for other_img_gt in images_gt_list:
            if this_img_gt.id == other_img_gt.id:
                continue
            
            other_name = other_img_gt.name.split('/')[-1]
            if other_name not in images_by_name:
                dts.append(np.inf)
                dRs.append(180)
                continue
            
            other_img = images_by_name[other_name]
            
            # Compute relative poses
            R_rel, t_rel = compute_relative_pose(this_img, other_img)
            R_rel_gt, t_rel_gt = compute_relative_pose(this_img_gt, other_img_gt)
            
            # Compute error between estimated and GT relative pose
            R_error = R_rel @ R_rel_gt.T
            
            # Translation angular error
            if np.linalg.norm(t_rel_gt) < min_proj_center_dist:
                dt = 0
            else:
                t1 = t_rel / max(1e-10, np.linalg.norm(t_rel))
                t2 = t_rel_gt / max(1e-10, np.linalg.norm(t_rel_gt))
                cos_dist = np.clip(np.dot(t1, t2), -1, 1)
                dt = np.rad2deg(np.arccos(cos_dist))
            
            dR = rotation_matrix_angle(R_error)
            dts.append(dt)
            dRs.append(dR)
    
    return np.array(dts), np.array(dRs)

def compute_abs_pose_errors(images_gt, images):
    """Compute absolute pose errors after alignment.
    
    Args:
        images_gt: dict of GT images (image_id -> Image)
        images: dict of reconstructed images (image_id -> Image)
    """
    images_gt_list = list(images_gt.values())
    dts = np.full(len(images_gt_list), np.inf, dtype=np.float64)
    dRs = np.full(len(images_gt_list), 180, dtype=np.float64)
    
    if images is None or len(images) == 0:
        return dts, dRs
    
    images_by_name = {img.name.split('/')[-1]: img for img in images.values()}
    
    for i, img_gt in enumerate(images_gt_list):
        img_gt_name = img_gt.name.split('/')[-1]
        if img_gt_name not in images_by_name:
            continue
        
        img = images_by_name[img_gt_name]
        
        # Compute pose error: estimated_from_gt
        R_gt = qvec2rotmat(img_gt.qvec)
        R_est = qvec2rotmat(img.qvec)
        R_error = R_est @ R_gt.T
        
        t_error = img.tvec - img_gt.tvec
        dts[i] = np.linalg.norm(t_error)
        dRs[i] = rotation_matrix_angle(R_error)
    
    return dts, dRs

def compute_auc(errors, thresholds, min_error=0.0):
    """Compute AUC (Area Under Curve) at given thresholds."""
    if len(errors) == 0:
        return np.zeros(len(thresholds))
    
    errors = np.sort(errors)
    recall = (np.arange(len(errors)) + 1) / len(errors)
    
    if min_error > 0:
        min_index = np.searchsorted(errors, min_error, side="right")
        min_score = min_index / len(errors)
        recall = np.r_[min_score, min_score, recall[min_index:]]
        errors = np.r_[0, min_error, errors[min_index:]]
    else:
        recall = np.r_[0, recall]
        errors = np.r_[0, errors]
    
    aucs = np.zeros(len(thresholds), dtype=np.float64)
    for i, t in enumerate(thresholds):
        last_index = np.searchsorted(errors, t, side="right")
        r = np.r_[recall[:last_index], recall[last_index - 1]]
        e = np.r_[errors[:last_index], t]
        # Use np.trapz for compatibility with older numpy versions
        auc = np.trapz(r, x=e) / t
        aucs[i] = auc * 100
    return aucs / 1.1

def metric_pose_auc(runner_result, dataset_path, error_type="relative", thresholds=None, **kwargs):
    """Compute pose AUC metric (relative or absolute pose errors)."""
    model = runner_result.get("model_path")
    if not model:
        return None
    
    # Find GT sparse model
    sparse_gt_path = os.path.join(dataset_path, "sparse_gt", "0")
    if not os.path.exists(sparse_gt_path):
        return None
    
    # Load GT images
    images_gt = load_images_from_model(sparse_gt_path)
    if images_gt is None:
        return None
    
    # Load reconstructed model
    if error_type == "absolute":
        # Use aligned model for absolute errors
        ref_pose_txt = None
        if os.path.exists(os.path.join(sparse_gt_path, "images.bin")):
            ref_pose_txt = extract_poses_from_images_bin(sparse_gt_path)
        elif os.path.exists(os.path.join(sparse_gt_path, "images.txt")):
            ref_pose_txt = extract_poses_from_images_txt(sparse_gt_path)
        if ref_pose_txt is None and os.path.exists(os.path.join(dataset_path, "pose_merged.txt")):
            ref_pose_txt = os.path.join(dataset_path, "pose_merged.txt")
        
        if ref_pose_txt:
            model = align_model(model, ref_pose_txt, alignment_max_error=4.0)
    
    # Load reconstructed images
    images = load_images_from_model(model)
    if images is None:
        return None
    
    # Compute errors
    if error_type == "relative":
        if thresholds is None:
            thresholds = [1, 3, 5, 10]  # degrees
        dts, dRs = compute_rel_pose_errors(images_gt, images, min_proj_center_dist=0.1)
        errors = np.maximum(dts, dRs)  # Use maximum of translation and rotation error
    elif error_type == "absolute":
        if thresholds is None:
            thresholds = [0.02, 0.05, 0.2, 0.5]  # meters
        dts, dRs = compute_abs_pose_errors(images_gt, images)
        errors = dts  # Use translation error for absolute
    else:
        return None
    
    aucs = compute_auc(errors, np.array(thresholds), min_error=0.0)
    # Return mean AUC across all thresholds
    return float(np.mean(aucs))

def metric_rotation_error_abs(runner_result, dataset_path, **kwargs):
    """Compute absolute rotation error (degrees) for registered images after alignment."""
    model = runner_result.get("model_path")
    if not model:
        return None
    
    # Find GT sparse model
    sparse_gt_path = os.path.join(dataset_path, "sparse_gt", "0")
    if not os.path.exists(sparse_gt_path):
        sparse_gt_path = os.path.join(dataset_path, "sparse_gt")
    if not os.path.exists(sparse_gt_path):
        return None
    
    # Align model first for absolute errors
    ref_pose_txt = None
    if os.path.exists(os.path.join(sparse_gt_path, "images.bin")):
        ref_pose_txt = extract_poses_from_images_bin(sparse_gt_path)
    elif os.path.exists(os.path.join(sparse_gt_path, "images.txt")):
        ref_pose_txt = extract_poses_from_images_txt(sparse_gt_path)
    if ref_pose_txt is None and os.path.exists(os.path.join(dataset_path, "pose_merged.txt")):
        ref_pose_txt = os.path.join(dataset_path, "pose_merged.txt")
    
    if ref_pose_txt:
        model = align_model(model, ref_pose_txt, alignment_max_error=4.0)
    
    # Load images
    images_gt = load_images_from_model(sparse_gt_path)
    images = load_images_from_model(model)
    if images_gt is None or images is None:
        return None
    
    _, dRs = compute_abs_pose_errors(images_gt, images)
    valid_errors = dRs[dRs < 180]
    
    if len(valid_errors) == 0:
        return None
    return float(np.median(valid_errors))

def metric_translation_error_abs(runner_result, dataset_path, **kwargs):
    """Compute absolute translation error (meters) for registered images after alignment."""
    model = runner_result.get("model_path")
    if not model:
        return None
    
    # Find GT sparse model
    sparse_gt_path = os.path.join(dataset_path, "sparse_gt", "0")
    if not os.path.exists(sparse_gt_path):
        sparse_gt_path = os.path.join(dataset_path, "sparse_gt")
    if not os.path.exists(sparse_gt_path):
        return None
    
    # Align model first for absolute errors
    ref_pose_txt = None
    if os.path.exists(os.path.join(sparse_gt_path, "images.bin")):
        ref_pose_txt = extract_poses_from_images_bin(sparse_gt_path)
    elif os.path.exists(os.path.join(sparse_gt_path, "images.txt")):
        ref_pose_txt = extract_poses_from_images_txt(sparse_gt_path)
    if ref_pose_txt is None and os.path.exists(os.path.join(dataset_path, "pose_merged.txt")):
        ref_pose_txt = os.path.join(dataset_path, "pose_merged.txt")
    
    if ref_pose_txt:
        model = align_model(model, ref_pose_txt, alignment_max_error=4.0)
    
    # Load images
    images_gt = load_images_from_model(sparse_gt_path)
    images = load_images_from_model(model)
    if images_gt is None or images is None:
        return None
    
    dts, _ = compute_abs_pose_errors(images_gt, images)
    valid_errors = dts[dts < np.inf]
    
    if len(valid_errors) == 0:
        return None
    return float(np.median(valid_errors))

def metric_rotation_error_rel(runner_result, dataset_path, **kwargs):
    """Compute relative rotation error (degrees) between image pairs."""
    model = runner_result.get("model_path")
    if not model:
        return None
    
    # Find GT sparse model
    sparse_gt_path = os.path.join(dataset_path, "sparse_gt", "0")
    if not os.path.exists(sparse_gt_path):
        sparse_gt_path = os.path.join(dataset_path, "sparse_gt")
    if not os.path.exists(sparse_gt_path):
        return None
    
    images_gt = load_images_from_model(sparse_gt_path)
    images = load_images_from_model(model)
    if images_gt is None or images is None:
        return None
    
    dts, dRs = compute_rel_pose_errors(images_gt, images, min_proj_center_dist=0.1)
    return float(np.median(dRs))

def metric_translation_error_rel(runner_result, dataset_path, **kwargs):
    """Compute relative translation angular error (degrees) between image pairs after alignment."""
    model = runner_result.get("model_path")
    if not model:
        return None
    
    # Find GT sparse model
    sparse_gt_path = os.path.join(dataset_path, "sparse_gt", "0")
    if not os.path.exists(sparse_gt_path):
        sparse_gt_path = os.path.join(dataset_path, "sparse_gt")
    if not os.path.exists(sparse_gt_path):
        return None
    
    # Align model first (relative errors also need alignment)
    ref_pose_txt = None
    if os.path.exists(os.path.join(sparse_gt_path, "images.bin")):
        ref_pose_txt = extract_poses_from_images_bin(sparse_gt_path)
    elif os.path.exists(os.path.join(sparse_gt_path, "images.txt")):
        ref_pose_txt = extract_poses_from_images_txt(sparse_gt_path)
    if ref_pose_txt is None and os.path.exists(os.path.join(dataset_path, "pose_merged.txt")):
        ref_pose_txt = os.path.join(dataset_path, "pose_merged.txt")
    
    if ref_pose_txt:
        model = align_model(model, ref_pose_txt, alignment_max_error=4.0)
    
    # Load images from aligned model
    images_gt = load_images_from_model(sparse_gt_path)
    images = load_images_from_model(model)
    if images_gt is None or images is None:
        return None
    
    dts, dRs = compute_rel_pose_errors(images_gt, images, min_proj_center_dist=0.1)
    valid_errors = dts[dts < np.inf]
    
    if len(valid_errors) == 0:
        return None
    return float(np.median(valid_errors))

def metric_registration_ratio(runner_result, dataset_path, **kwargs):
    """Compute ratio of registered images to total images."""
    model = runner_result.get("model_path")
    if not model:
        return 0.0
    
    # Find GT sparse model
    sparse_gt_path = os.path.join(dataset_path, "sparse_gt", "0")
    if not os.path.exists(sparse_gt_path):
        sparse_gt_path = os.path.join(dataset_path, "sparse_gt")
    if not os.path.exists(sparse_gt_path):
        return None
    
    # Load images
    images_gt = load_images_from_model(sparse_gt_path)
    images = load_images_from_model(model)
    if images_gt is None or images is None:
        return 0.0
    
    images_by_name = {img.name.split('/')[-1]: img for img in images.values()}
    images_gt_list = list(images_gt.values())
    num_registered = sum(1 for img_gt in images_gt_list if img_gt.name.split('/')[-1] in images_by_name)
    return num_registered / len(images_gt_list)


# ---------------- scale metrics ----------------

def _get_camera_centers(images_dict):
    """Extract camera centers (world positions) from images dict.

    In COLMAP convention: X_world = -R^T @ t
    
    Returns:
        centers: (N, 3) array of camera center positions in world frame
        names: list of image names (basename)
    """
    centers = []
    names = []
    for img in images_dict.values():
        R = qvec2rotmat(img.qvec)
        t = img.tvec
        c = -R.T @ t
        centers.append(c)
        names.append(img.name.split('/')[-1])
    return np.array(centers), names


def _find_gt_sparse(dataset_path):
    """Return path to GT sparse model directory, or None."""
    for candidate in [
        os.path.join(dataset_path, "sparse_gt", "0"),
        os.path.join(dataset_path, "sparse_gt"),
    ]:
        if os.path.exists(candidate):
            return candidate
    return None


def _load_matched_centers(images_gt, images_est):
    """Return paired GT and estimated camera centers for commonly registered images.

    Returns:
        centers_gt:  (N, 3)
        centers_est: (N, 3)
    """
    est_by_name = {img.name.split('/')[-1]: img for img in images_est.values()}
    gt_list, est_list = [], []
    for img_gt in images_gt.values():
        name = img_gt.name.split('/')[-1]
        if name in est_by_name:
            R_gt = qvec2rotmat(img_gt.qvec)
            gt_list.append(-R_gt.T @ img_gt.tvec)
            R_est = qvec2rotmat(est_by_name[name].qvec)
            est_list.append(-R_est.T @ est_by_name[name].tvec)
    if len(gt_list) < 2:
        return None, None
    return np.array(gt_list), np.array(est_list)


def estimate_scale_factor(centers_gt, centers_est):
    """Estimate the scale factor s such that ||est|| ≈ s * ||gt||.

    Uses the ratio of pairwise inter-camera distances (median for robustness).
    
    Returns:
        s: estimated scale factor (est / gt), > 1 means over-estimated scale
    """
    n = len(centers_gt)
    if n < 2:
        return None
    # Sample O(n) pairs to keep cost reasonable for large reconstructions
    rng = np.random.default_rng(0)
    idx = np.arange(n)
    if n > 200:
        idx = rng.choice(n, 200, replace=False)
    gt_sub = centers_gt[idx]
    est_sub = centers_est[idx]

    ratios = []
    for i in range(len(idx)):
        for j in range(i + 1, len(idx)):
            d_gt = np.linalg.norm(gt_sub[i] - gt_sub[j])
            d_est = np.linalg.norm(est_sub[i] - est_sub[j])
            if d_gt > 1e-6:
                ratios.append(d_est / d_gt)
    if not ratios:
        return None
    return float(np.median(ratios))


def metric_scale_ratio(runner_result, dataset_path, **kwargs):
    """Scale ratio: estimated_scale / gt_scale (trajectory length ratio).

    Ideal value = 1.0.  A value of 2.0 means the reconstruction is 2× too large.
    Computed as median pairwise inter-camera distance ratio (est / gt).
    """
    model = runner_result.get("model_path")
    if not model:
        return None
    sparse_gt_path = _find_gt_sparse(dataset_path)
    if sparse_gt_path is None:
        return None

    images_gt = load_images_from_model(sparse_gt_path)
    images_est = load_images_from_model(model)
    if images_gt is None or images_est is None:
        return None

    centers_gt, centers_est = _load_matched_centers(images_gt, images_est)
    if centers_gt is None:
        return None

    s = estimate_scale_factor(centers_gt, centers_est)
    return s


METRICS = {
    "timing": metric_timing,
    "chamfer": metric_chamfer,
    "points": metric_points_count,
    # Pose AUC metrics
    "pose_auc": metric_pose_auc,  # relative pose AUC (default)
    "pose_auc_rel": metric_pose_auc,  # relative pose AUC (explicit)
    "pose_auc_abs": lambda result, dataset_path, **kwargs: metric_pose_auc(result, dataset_path, error_type="absolute", **kwargs),
    # Absolute pose errors (after alignment)
    "rotation_error_abs": metric_rotation_error_abs,
    "translation_error_abs": metric_translation_error_abs,
    # Relative pose errors (alignment-invariant)
    "rotation_error_rel": metric_rotation_error_rel,
    "translation_error_rel": metric_translation_error_rel,
    # Default aliases
    "rotation_error": metric_rotation_error_rel,
    "translation_error": metric_translation_error_rel,
    # Registration ratio
    "registration_ratio": metric_registration_ratio,
    # Scale consistency metrics
    "scale_ratio": metric_scale_ratio,              # est/gt scale ratio, ideal=1.0
}

# ---------------- orchestrator ----------------
def run_one_scene(dataset_key, dataset_path, scene, method_key, metrics, out_csv, options):
    is_multi_camera_rig = options.get("is_multi_camera_rig", False)
    scene_root, img_path, db_path = get_scene_paths(dataset_key, dataset_path, scene, is_multi_camera_rig)

    skip_reconstruction = options.get("skip_reconstruction", False)
    
    if skip_reconstruction:
        # Skip reconstruction, just locate existing model
        print(f"Skipping reconstruction for {dataset_key}/{scene}/{method_key}, using existing results")
        
        # Determine output directory based on method
        if method_key == "colmap":
            out_dir = os.path.join(scene_root, "sparse_colmap")
        elif method_key == "glomap":
            out_dir = os.path.join(scene_root, "sparse_glomap")
        elif method_key == "instantsfm":
            out_dir = os.path.join(scene_root, "sparse")
        else:
            out_dir = os.path.join(scene_root, f"sparse_{method_key}")
        
        model0 = os.path.join(out_dir, "0")
        if os.path.exists(model0):
            result = {"model_path": model0, "time": None}
        elif os.path.exists(out_dir):
            result = {"model_path": out_dir, "time": None}
        else:
            print(f"Warning: No existing reconstruction found at {out_dir}")
            result = {"model_path": None, "time": None, "error": "No existing reconstruction found"}
    else:
        # ensure database (feature extraction) if required
        recompute_db = options.get("recompute_database", False)
        ok_db = ensure_database(scene_root, img_path, db_path, force_recompute=recompute_db, is_multi_camera_rig=is_multi_camera_rig)
        if not ok_db:
            print(f"Warning: database not ready for {scene_root}, proceeding may fail.")

        # call runner with scene_root as dataset_path (runners expect dataset_path to contain images/database)
        runner = RUNNERS.get(method_key)
        if runner is None:
            print(f"Unknown method {method_key}")
            return {"error": f"Unknown method {method_key}"}

        try:
            result = runner(scene_root, scene, options.get("runner_options", {}))
        except Exception as e:
            print(f"Error running {method_key} for {dataset_key}/{scene}: {e}")
            import traceback
            traceback.print_exc()
            result = {"model_path": None, "time": None, "error": str(e)}

    metric_values = {}
    for m in metrics:
        try:
            metric_values[m] = METRICS[m](result, dataset_path=scene_root)
        except Exception as e:
            print(f"Error computing metric {m} for {dataset_key}/{scene} {method_key}: {e}")
            # Return default maximum error value for missing/failed metrics
            # For error metrics, inf means worst case; for other metrics, None
            if 'error' in m or 'chamfer' in m:
                metric_values[m] = float('inf')
            elif 'ratio' in m or 'auc' in m:
                metric_values[m] = 0.0
            else:
                metric_values[m] = None

    row = [dataset_key, scene, method_key] + [metric_values.get(m) for m in metrics]
    with open(out_csv, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(row)

    return {"dataset": dataset_key, "scene": scene, "method": method_key, "metrics": metric_values, "runner": result}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=["360"], help="dataset keys, e.g. 360 ScanNet")
    parser.add_argument("--methods", nargs="+", default=["instantsfm"], help="methods: colmap glomap instantsfm")
    parser.add_argument("--metrics", nargs="+", default=["timing"], 
                        help="metrics: timing chamfer points pose_auc pose_auc_abs rotation_error translation_error "
                             "registration_ratio scale_ratio")
    parser.add_argument("--out", default="results.csv")
    parser.add_argument("--recompute-database", action="store_true", help="force re-run COLMAP feature extraction & matching per scene")
    parser.add_argument("--skip-reconstruction", action="store_true", help="skip reconstruction phase, only evaluate existing results")
    args = parser.parse_args()

    # header: dataset, scene, method, <metrics...>
    header = ["dataset", "scene", "method"] + args.metrics
    with open(args.out, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)

    options = {
        "recompute_database": args.recompute_database,
        "skip_reconstruction": args.skip_reconstruction
    }
    for d in args.datasets:
        ds = DATASET_CONFIGS.get(d)
        if ds is None:
            print(f"Unknown dataset {d}, skipping")
            continue
        dataset_path = ds["path"]
        is_multi_camera_rig = ds.get("multi_camera_rig", False)
        scenes = list_scenes_for_dataset(d, dataset_path)
        if len(scenes) == 0:
            print(f"No scenes found for dataset {d} at {dataset_path}, skipping")
            continue

        for scene in scenes:
            # Update options with dataset-specific settings
            scene_options = options.copy()
            scene_options["is_multi_camera_rig"] = is_multi_camera_rig
            
            for m in args.methods:
                print(f"Running: dataset={d}, scene={scene}, method={m}")
                res = run_one_scene(d, dataset_path, scene, m, args.metrics, args.out, scene_options)
                print("done:", res)

if __name__ == "__main__":
    main()