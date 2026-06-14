import os
import subprocess
import time
import glob

from instantsfm.utils.read_write_model import read_images_binary

def extract_poses(model_path):
    pose_lines = []

    cam_files = sorted(glob.glob(os.path.join(model_path, "*_cam.txt")))
    for cam_file in cam_files:
        basename = os.path.basename(cam_file)
        num = basename.split('_')[0]
        with open(cam_file, 'r') as f:
            lines = f.readlines()

        extrinsic_lines = []
        extrinsic_found = False
        for line in lines:
            if line.strip() == "extrinsic":
                extrinsic_found = True
                continue
            if extrinsic_found and len(extrinsic_lines) < 4:
                extrinsic_lines.append(line.strip())
        if len(extrinsic_lines) < 4:
            print(f"Warning: {cam_file} extrinsic matrix not found or incomplete.")
            continue

        translation = []
        for i in range(3):
            vals = extrinsic_lines[i].split()
            translation.append(vals[3])

        pose_lines.append(f"{num}.jpg {translation[0]} {translation[1]} {translation[2]}")

    output_path = os.path.join(model_path, "pose_merged.txt")
    with open(output_path, "w") as f:
        for line in pose_lines:
            f.write(line + "\n")
    print(f"Saved merged poses to {output_path}")


def align_and_convert_model(input_path, output_path, dataset_path):
    aligned_path = f"{output_path}_aligned"
    if os.path.exists(aligned_path):
        subprocess.run(["rm", "-rf", aligned_path])
    os.makedirs(aligned_path, exist_ok=True)

    align_command = [
        "colmap", "model_aligner",
        "--input_path", input_path,
        "--output_path", aligned_path,
        "--ref_is_gps", "0",
        "--ref_images_path", f"{dataset_path}/pose_merged.txt",
        "--alignment_max_error", "3.0"
    ]
    
    try:
        subprocess.run(align_command, check=True)
        print(f"Successfully aligned model at {aligned_path}")
    except subprocess.CalledProcessError as e:
        print(f"Alignment failed: {e}")
        subprocess.run(["cp", "-r", input_path, aligned_path])

    return aligned_path

datasets_intermediate = ["Family", "Francis", "Horse", "Lighthouse", "M60", "Panther", "Playground", "Train"]
datasets_advanced = ["Auditorium", "Ballroom", "Courtroom", "Museum", "Palace", "Temple"]
datasets = datasets_intermediate  # + datasets_advanced

timing_results = {}

'''for idx, dataset in enumerate(datasets):
    print(f"Processing dataset: {dataset}")  
    dataset_path = f"dataset/tt/Intermediate/{dataset}"
    # feature handling
    feature_extraction_command = f"colmap feature_extractor --database_path {dataset_path}/database.db --image_path {dataset_path}/images"
    subprocess.run(feature_extraction_command.split())
    feature_matching_command = f"colmap exhaustive_matcher --database_path {dataset_path}/database.db"
    subprocess.run(feature_matching_command.split())
    
    # colmap processing
    if os.path.exists(f"{dataset_path}/sparse_colmap"):
        subprocess.run(["rm", "-rf", f"{dataset_path}/sparse_colmap"])
    start_time = time.time()
    os.makedirs(f"{dataset_path}/sparse_colmap", exist_ok=True)
    colmap_command = f"colmap mapper --database_path {dataset_path}/database.db --image_path {dataset_path}/images --output_path {dataset_path}/sparse_colmap"
    subprocess.run(colmap_command.split())
    colmap_time = time.time() - start_time

    # glomap processing
    start_time = time.time()
    if os.path.exists(f"{dataset_path}/sparse_glomap"):
        subprocess.run(["rm", "-rf", f"{dataset_path}/sparse_glomap"])
    os.makedirs(f"{dataset_path}/sparse_glomap", exist_ok=True)
    glomap_command = f"glomap mapper --database_path {dataset_path}/database.db --image_path {dataset_path}/images --output_path {dataset_path}/sparse_glomap"
    subprocess.run(glomap_command.split())
    glomap_time = time.time() - start_time

    # instantsfm processing
    start_time = time.time()
    if os.path.exists(f"{dataset_path}/sparse"):
        subprocess.run(["rm", "-rf", f"{dataset_path}/sparse"])
    os.makedirs(f"{dataset_path}/sparse", exist_ok=True)
    subprocess.run(["ins-sfm", "--data", f"{dataset_path}"])
    instantsfm_time = time.time() - start_time

    timing_results[dataset] = {
        "colmap_time": colmap_time,
        "glomap_time": glomap_time,
        "instantsfm_time": instantsfm_time
    }

    extract_poses(f"{dataset_path}/cams_1/")
    align_and_convert_model(f"{dataset_path}/sparse_colmap/0", f"{dataset_path}/sparse_colmap/0", f"{dataset_path}/cams_1")
    align_and_convert_model(f"{dataset_path}/sparse_glomap/0", f"{dataset_path}/sparse_glomap/0", f"{dataset_path}/cams_1")
    align_and_convert_model(f"{dataset_path}/sparse/0", f"{dataset_path}/sparse/0", f"{dataset_path}/cams_1")


for dataset, times in timing_results.items():
    print(f"Dataset: {dataset}")
    print(f"  COLMAP time: {times['colmap_time']:.2f} seconds")
    print(f"  GloMap time: {times['glomap_time']:.2f} seconds")
    print(f"  InstantSfM time: {times['instantsfm_time']:.2f} seconds")
    print()'''

for idx, dataset in enumerate(datasets):
    print(f"Processing dataset: {dataset}")  
    dataset_path = f"dataset/tt/Intermediate/{dataset}"
    # instantsfm processing
    start_time = time.time()
    if os.path.exists(f"{dataset_path}/sparse"):
        subprocess.run(["rm", "-rf", f"{dataset_path}/sparse"])
    os.makedirs(f"{dataset_path}/sparse", exist_ok=True)
    subprocess.run(["ins-sfm", "--data", f"{dataset_path}"])
    instantsfm_time = time.time() - start_time

    timing_results[dataset] = {
        "instantsfm_time": instantsfm_time
    }


for dataset, times in timing_results.items():
    print(f"Dataset: {dataset}")
    print(f"  InstantSfM time: {times['instantsfm_time']:.2f} seconds")
    print()