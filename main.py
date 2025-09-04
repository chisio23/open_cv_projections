import numpy as np
import cv2
import json
import csv
import matplotlib.pyplot as plt

cameras_txt_path = "data/cameras/cameras.txt"

def load_cam_intrinsics(cameras_txt_path, camera_id, target_size=None):
    
    with open(cameras_txt_path, "r") as f:
        
        for line in f:
            if line.startswith("#") or line.strip() == "":
                continue
            
            parts = line.strip().split()
            
            cam_id = int(parts[0])
                      
            if cam_id == camera_id:
                
                model = parts[1]
                width = int(parts[2])
                height = int(parts[3])
                fx = float(parts[4])
                fy = float(parts[5])
                cx = float(parts[6])
                cy = float(parts[7])
                             
                return {
                    "fx": fx,
                    "fy": fy,
                    "cx": cx,
                    "cy": cy,
                    "width": width,
                    "height": height,
                    "model": model,
                }                    
    pass


def load_image(image_path = "data/skin_color_1/cam_0000/0000.png"):
    
    image_open = cv2.imread(image_path, cv2.IMREAD_COLOR)
    
    if image_open is None:
        print("Error: Image not found or unable to read.")
        
    else: 
        print("Image loaded successfully!")
        
    height, width, channels = image_open.shape
    
    return{
        
        "width": width,
        "height": height,
        "channels": channels
        
    }          
    


def load_extrinsics(extrinsics_source, frame_id, input_convention="cw"):
    """
    Load extrinsics for the frame.
    Inputs:
        extrinsics_source: path or object containing extrinsics
        frame_id: which frame to use
        input_convention: "cw" (camera->world) or "wc" (world->camera)
    Returns:
        dict with R_wc (3x3), t_wc (3x1)
    """
    pass


def parse_skl(skl_path):
    """
    Parse skeleton file (.skl).
    Inputs:
        skl_path: path to skl file
    Returns:
        joint_names: list of strings
        parents: list of ints (-1 for root)
        joints_raw: Nx3 numpy array
        units_hint: string ("m", "cm", etc.)
    """
    pass


def infer_global_or_local(parents, joints_raw):
    """
    Heuristic to check if joints are global or local positions.
    Returns:
        dict with {"are_global": bool, "scale_suggestion": float or None}
    """
    pass


def to_world_positions(joint_names, parents, joints_raw, are_global=True, root_transform=None):
    """
    Convert local skeleton positions to world positions if needed.
    Returns:
        Nx3 numpy array in world coordinates
    """
    pass


def apply_unit_scale(joints_world, units_hint="m"):
    """
    Convert to meters.
    Returns:
        joints_m, scale_used
    """
    pass


def world_to_camera(joints_m, extrinsics):
    """
    Transform world points into camera coordinates.
    Returns:
        Xc (Nx3), Zc (N,)
    """
    pass


def project_pinhole(Xc, intrinsics):
    """
    Pinhole projection without distortion.
    Returns:
        uv (Nx2) pixel coordinates
    """
    pass


def visibility_mask(uv, Zc, width, height):
    """
    Check which joints are visible.
    Returns:
        boolean mask (N,)
    """
    pass


def build_edges_from_parents(parents):
    """
    Build list of bones as (parent, child) pairs.
    """
    pass


def sort_edges_by_depth(edges, Zc, order="far_to_near"):
    """
    Sort edges by average depth.
    """
    pass


def draw_skeleton(image, uv, visible_mask, edges_sorted, style=None):
    """
    Draw joints and bones on the image.
    Returns:
        overlay image
    """
    pass


def export_results(out_dir, overlay_image, uv, visible_mask, Zc, joint_names, meta):
    """
    Save overlay image and 2D keypoints/logs.
    """
    pass


def main():
    
    load_cam_intrinsics(cameras_txt_path, 1)
    load_image()
    
    """
    Orchestrate the full pipeline for one frame.
    Steps:
      1. Load intrinsics
      2. Load image
      3. Load extrinsics
      4. Parse skeleton
      5. Normalize skeleton to world + meters
      6. Transform to camera
      7. Project to 2D
      8. Check visibility
      9. Build & sort edges
     10. Draw overlay
     11. Export results
    """
    pass


if __name__ == "__main__":
    main()
