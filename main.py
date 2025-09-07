import numpy as np
import cv2
import json
import csv
import matplotlib.pyplot as plt
import os

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
    # Minimal fix: fail early if OpenCV can't read the image (avoid .shape on None)
    image_open = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image_open is None:
        raise FileNotFoundError(f"Error: Image not found or unable to read → {image_path}")
    else: 
        print("Image loaded successfully!")
        
    height, width, channels = image_open.shape
    
    return {
    "width": width,
    "height": height,
    "channels": channels,
    "image": image_open
    }  
            
def load_extrinsics_by_name(image_name, extrinsics_source="data/cameras/images.txt"):
    target = os.path.basename(image_name)  # <<< only take filename
    with open(extrinsics_source, "r") as f:
        for line in f:
            if line.startswith("#") or line.strip() == "":
                continue
            parts = line.strip().split()
            # IMAGE_ID qw qx qy qz tx ty tz CAMERA_ID IMAGE_NAME
            if os.path.basename(parts[-1]) == target:
                qw = float(parts[1]); qx = float(parts[2]); qy = float(parts[3]); qz = float(parts[4])
                tx = float(parts[5]); ty = float(parts[6]); tz = float(parts[7])
                cam_id = int(parts[8])

                n = (qw*qw + qx*qx + qy*qy + qz*qz) ** 0.5
                qw, qx, qy, qz = qw/n, qx/n, qy/n, qz/n
                R = np.array([
                    [1-2*(qy*qy+qz*qz),   2*(qx*qy - qz*qw), 2*(qx*qz + qy*qw)],
                    [2*(qx*qy + qz*qw),   1-2*(qx*qx+qz*qz), 2*(qy*qz - qx*qw)],
                    [2*(qx*qz - qy*qw),   2*(qy*qz + qx*qw), 1-2*(qx*qx+qy*qy)]
                ], dtype=float)
                t = np.array([tx, ty, tz], dtype=float)
                return {"R": R, "T": t, "camera_id": cam_id, "image_name": target}
    raise ValueError(f"Image name '{image_name}' not found in images.txt")         
    
def parse_skl(skl_path="data/skeleton/0000.skl"):
    joint_names = []
    parent_names = []   # guardamos los nombres de padres como strings
    joint_raw = []

    with open(skl_path, "r") as f:
        for line in f:
            if line.startswith("#") or line.strip() == "":
                continue

            parts = line.strip().split()

            # saltar cabeceras
            if len(parts) < 5 or parts[2].lower() in {"pos_x", "px"}:
                continue

            joint_name = parts[0]
            parent_name = parts[1]

            try:
                pos_x = float(parts[2])
                pos_y = float(parts[3])
                pos_z = float(parts[4])
            except ValueError:
                continue  # no se puede convertir, saltamos

            # rotaciones opcionales
            rot_x = rot_y = rot_z = rot_w = None
            if len(parts) >= 9:
                rot_x = float(parts[5])
                rot_y = float(parts[6])
                rot_z = float(parts[7])
                rot_w = float(parts[8])

            pos_vec = np.array([pos_x, pos_y, pos_z], dtype=float)

            joint_names.append(joint_name)
            parent_names.append(parent_name)
            joint_raw.append(pos_vec)

    # convertir a np.array
    joints_raw = np.array(joint_raw, dtype=float)

    # construir diccionario nombre → índice
    name_to_idx = {name: idx for idx, name in enumerate(joint_names)}

    # traducir los parent_names a índices
    parents = []
    for pname in parent_names:
        if pname.lower() in {"root", "none", "-1"}:
            parents.append(-1)
        else:
            parents.append(name_to_idx[pname])

    units_hint = "m"  # provisional
    return joint_names, parents, joints_raw, units_hint



def infer_global_or_local(parents, joints_raw):
    bone_lengths = []
    for i, p in enumerate(parents):
        if p >= 0:
            bl = float(np.linalg.norm(joints_raw[i] - joints_raw[p]))
            if np.isfinite(bl) and bl > 0:
                bone_lengths.append(bl)

    if not bone_lengths:
        print(" No bone lengths found")
        return {"are_global": False, "scale_suggestion": None}

    median_len = float(np.median(bone_lengths))
    if median_len <= 0 or not np.isfinite(median_len):
        print(" Invalid median bone length")
        return {"are_global": False, "scale_suggestion": None}

    # Scale suggestion
    if 0.2 <= median_len <= 0.6:         # meters
        scale = 1.0
        scale_str = "meters"
    elif 20.0 <= median_len <= 60.0:     # centimeters
        scale = 0.01
        scale_str = "centimeters"
    elif 200.0 <= median_len <= 600.0:   # millimeters
        scale = 0.001
        scale_str = "millimeters"
    else:
        scale = None
        scale_str = "unknown units"

    # Global vs local
    centroid = np.mean(joints_raw, axis=0)
    spread = float(np.median(np.linalg.norm(joints_raw - centroid, axis=1)))
    ratio = spread / median_len if median_len > 0 else float("inf")
    are_global = ratio >= 2.5

    # Debug print
    gtype = "GLOBAL" if are_global else "LOCAL"
    print(f"Skeleton type: {gtype}, median bone = {median_len:.3f}, units ~ {scale_str}")

    return {"are_global": are_global, "scale_suggestion": scale}

                

# def to_world_positions(joint_names, parents, joints_raw, are_global=True, root_transform=None):
#     """
#     Convert local skeleton positions to world positions if needed.
#     Returns:
#         Nx3 numpy array in world coordinates
#     """
#     pass


def apply_unit_scale(joints_world, units_hint="m"):
    
    
    if units_hint == "m":
        scale = 1.0
    elif units_hint == "cm":
        scale = 0.01
    elif units_hint == "mm":
        scale = 0.001
    else:
        scale = 1.0
    joints_m = joints_world * scale
    return joints_m, scale


def world_to_camera(joints_m, extrinsics):
    
    xc_list = []
    zc_list = []
    
    for i in joints_m:
        
        xw = i
        
        R = extrinsics["R"]
        T = extrinsics["T"] 
        
        wc = R @ xw + T
        
        xc_list.append(wc)
        
    for z in xc_list:
        
        zc = z[2]
        
        zc_list.append(zc)
        
    return np.array(xc_list), np.array(zc_list)
        

def project_pinhole(xc_list, intrinsics):
    
    uv_list = []
    
    for i in xc_list: 
        
        fx = intrinsics["fx"]
        fy = intrinsics["fy"]
        cx = intrinsics["cx"]
        cy = intrinsics["cy"]
        
        
        k = np.array([
                    [fx, 0, cx],
                    [0, fy, cy],
                    [0, 0, 1]
                ])
        
        value = k @ i / i[2]
        
        uv_list.append([value[0], value[1]])
    
    return uv_list 


def visibility_mask(uv_list, zc_list, width, height):
    
    N = len(uv_list)
    return [True] * N


def build_edges_from_parents(parents):
    
    edges = []
    
    for i in range(len(parents)):
        
        p = parents[i]
        
        if p == -1: 
            
            continue
        
        else: 
                        
            edges.append((p, i))
            
    return edges

def sort_edges_by_depth(edges, zc_list, order="far_to_near"):
    
    pairs = []
    edges_sorted = []
    
    for (p, c) in edges:
        dpc = (zc_list[p] + zc_list[c]) / 2
        
        pairs.append((dpc, (p,c)))
        
    if order == "far_to_near":
        
       pairs.sort(key=lambda x: x[0], reverse=True) 
       
    else:
        
        pairs.sort(key=lambda x: x[0], reverse=False)
        
    edges_sorted = edges_sorted = [edge for (_, edge) in pairs]

    return edges_sorted   
    
def draw_skeleton(image, uv_list, visible_mask, edges_sorted, style=None):
    cfg = {
        "joint_radius": 3,
        "joint_color": (0, 255, 255),    # yellow
        "bone_color": (0, 255, 0),       # green
        "bone_thickness": 2,
        "alpha": 0.7,
        "line_type": cv2.LINE_AA,
    }
    if style:
        cfg.update(style)

    # Prepare data
    uv = np.asarray(uv_list, dtype=float)
    N = uv.shape[0]
    if visible_mask is None:
        mask = np.ones(N, dtype=bool)
    else:
        mask = np.asarray(visible_mask, dtype=bool)
        if mask.shape[0] != N:
            raise ValueError("visible_mask length does not match uv_list length.")

    H, W = image.shape[:2]
    def _in_bounds(pt):
        u, v = pt
        return (0 <= u < W) and (0 <= v < H)

    # Round to int for drawing (keep a safe copy)
    uv_int = np.round(uv).astype(int)

    base = image
    overlay = base.copy()

    # Draw bones (lines) first, in the provided order
    for (p, c) in edges_sorted:
        if p < 0 or c < 0 or p >= N or c >= N:
            continue
        if not (mask[p] and mask[c]):
            continue
        p_ok = np.isfinite(uv[p]).all() and _in_bounds(uv_int[p])
        c_ok = np.isfinite(uv[c]).all() and _in_bounds(uv_int[c])
        if not (p_ok and c_ok):
            continue
        pt1 = tuple(uv_int[p])
        pt2 = tuple(uv_int[c])
        cv2.line(
            overlay,
            pt1,
            pt2,
            color=cfg["bone_color"],
            thickness=cfg["bone_thickness"],
            lineType=cfg["line_type"],
        )

    # Draw joints (circles)
    for i in range(N):
        if not mask[i]:
            continue
        if not (np.isfinite(uv[i]).all() and _in_bounds(uv_int[i])):
            continue
        center = tuple(uv_int[i])
        cv2.circle(
            overlay,
            center,
            cfg["joint_radius"],
            cfg["joint_color"],
            thickness=-1,  # filled
            lineType=cfg["line_type"],
        )

    # Alpha blend for a nicer overlay
    alpha = float(cfg["alpha"])
    alpha = max(0.0, min(1.0, alpha))
    if alpha == 1.0:
        out = overlay
    elif alpha == 0.0:
        out = base.copy()
    else:
        out = cv2.addWeighted(overlay, alpha, base, 1.0 - alpha, 0.0)

    return out


def export_results(out_dir, overlay_image, uv_list, visible_mask, zc_list, joint_names, meta):
    """
    Save overlay image and 2D keypoints/logs.
    """
    pass

def _accumulate_local_to_world(parents, joints_local):
    J = np.array(joints_local, dtype=float).copy()
    N = len(J)
    for i in range(N):
        p = parents[i]
        if p >= 0:
            J[i] = J[p] + J[i]
    return J

def main():
    image_name = "cam_0000/0000.png"

    extrinsics = load_extrinsics_by_name(image_name, "data/cameras/images.txt")
    print("Extrinsics:", extrinsics)

    intrinsics = load_cam_intrinsics(cameras_txt_path, camera_id=extrinsics["camera_id"])
    print("Intrinsics:", intrinsics)

    img_info = load_image("data/skin_color_1/cam_0000/0000.png")
    print("Image:", img_info)

    joint_names, parents, joints_raw, units_hint = parse_skl("data/skeleton/0000.skl")
    print("Parsed skeleton:", len(joint_names), "joints; units_hint =", units_hint)

    info = infer_global_or_local(parents, joints_raw)
    print("Skeleton info:", info)

    joints_m, scale_used = apply_unit_scale(joints_raw, units_hint)
    print("Applied scale factor:", scale_used)

    Xc, Zc = world_to_camera(joints_m, extrinsics)
    print("Camera coords shape:", Xc.shape)
    print("Zc (first 5):", Zc[:5])

    image = img_info["image"]

    uv_list = project_pinhole(Xc, intrinsics)

    mask = visibility_mask(uv_list, Zc, intrinsics["width"], intrinsics["height"])

    edges = build_edges_from_parents(parents)
    edges_sorted = sort_edges_by_depth(edges, Zc, order="far_to_near")

    overlay_image = draw_skeleton(
        image, uv_list, mask, edges_sorted,
        style={"bone_color": (0,0,255), "joint_color": (255,255,255),
               "bone_thickness": 2, "joint_radius": 2, "alpha": 1.0}
    )
    cv2.imwrite("overlay_0000.png", overlay_image)
    cv2.imshow("Skeleton Overlay", overlay_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
        

if __name__ == "__main__":
    main()
