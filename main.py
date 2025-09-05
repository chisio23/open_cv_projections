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
            
def load_extrinsics(frame_id, extrinsics_source = "data/cameras/images.txt", input_convention="cw"):
        
    with open(extrinsics_source, "r") as f:
        
        for line in f:
            if line.startswith("#") or line.strip() == "":
                continue
            
            parts = line.strip().split()
            
            image_id = int(parts[0])
                      
            if image_id == frame_id:
                
                qw = float(parts[1])
                qx = float(parts[2])
                qy = float(parts[3])
                qz = float(parts[4])
                tx = float(parts[5])
                ty = float(parts[6])
                tz = float(parts[7])
                CAMERA_ID = int(parts[8])
                IMAGE_NAME = parts[9]
                
                norm = (qw**2 + qx**2 + qy**2 +qz**2) **0.5
                
                qw,qx,qy,qz = qw/norm, qx/norm, qy/norm, qz/norm
                
                R =  np.array([
                    [1-2*(qy**2 + qz**2), 2*(qx*qy - qz*qw), 2*(qx*qz + qy*qw)],
                    [2*(qx*qy + qz*qw), 1-2*(qx**2 + qz**2), 2*(qy*qz - qx*qw)],
                    [2*(qx*qz - qy*qw), 2*(qy*qz + qx*qw), 1-2*(qx**2 + qy**2)]
                ])
                            
                t = np.array([tx,ty,tz])
                             
                return {
                    
                    "R": R,
                    "T": t,
                    "camera_id": CAMERA_ID,
                    "image_name": IMAGE_NAME
                }              
    
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

    units_hint = "cm"  # provisional
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
        
    return xc_list, zc_list
        

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


def visibility_mask(uv_list, Zc, width, height):
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
    
    # 1) Intrínsecos (usa el camera_id que corresponda a tu frame)
    intrinsics = load_cam_intrinsics(cameras_txt_path, camera_id=1)
    print("Intrinsics:", intrinsics)

    # 2) Imagen
    img_info = load_image("data/skin_color_1/cam_0000/0000.png")
    print("Image:", img_info)

    # 3) Extrínsecos (usa el IMAGE_ID correcto del images.txt)
    extrinsics = load_extrinsics(frame_id=1, extrinsics_source="data/cameras/images.txt")
    print("Extrinsics:", extrinsics)

    # 4) Esqueleto
    joint_names, parents, joints_raw, units_hint = parse_skl("data/skeleton/0000.skl")
    print("Parsed skeleton:", len(joint_names), "joints; units_hint =", units_hint)

    # 5) Heurística global/local y escala sugerida (solo informativo)
    info = infer_global_or_local(parents, joints_raw)
    print("Skeleton info:", info)

    # 6) Escala a metros (puedes preferir info['scale_suggestion'] si no es None)
    units_for_scale = units_hint
    if info.get("scale_suggestion") is not None and units_hint == "m":
        # Si tu .skl no trae unidades fiables, puedes aplicar el sugerido
        print("Overriding units with heuristic scale suggestion.")
        # En ese caso, en vez de units_hint usarías un factor manual; aquí seguimos con units_hint.
    joints_m, scale_used = apply_unit_scale(joints_raw, units_for_scale)
    print("Applied scale factor:", scale_used)

    # 7) Mundo → Cámara
    Xc, Zc = world_to_camera(joints_m, extrinsics)
    print("Camera coords shape:", Xc.shape)
    print("Zc (first 5):", Zc[:5])
    
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
