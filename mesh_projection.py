import numpy as np
import cv2
import json
import csv
import matplotlib.pyplot as plt
import os
import random
import re 

# -----------------------------
# PATHS (EDIT THESE)
# -----------------------------
CAMERAS_TXT_PATH = "data/cameras/cameras.txt"
IMAGES_TXT_PATH  = "data/cameras/images.txt"
IMAGE_DIR        = "data/skin_color_1/cam_0000"
OBJ_DIR          = "data/skin_color_1/objs"

# IO
FRAME_GLOB_PATTERN = "frame_%04d.png"
OUTPUT_VIDEO_PATH  = "skeleton.mp4"

# RANGE
NUM_FRAMES = 57

cameras_txt_path = CAMERAS_TXT_PATH


def parse_mesh_obj(obj_path=OBJ_DIR + "/0000.obj"):
    
    mesh_vectors = []
    mesh_f = []
    
    with open(obj_path, "r") as f:
        for line in f:
            if not line.strip() or line.startswith("#"):
                continue

            if line.startswith("v "):
                parts = line.strip().split()
                x = float(parts[1]); y = float(parts[2]); z = float(parts[3])
                mesh_vectors.append([x, y, z])

            elif line.startswith("f "):
                parts = line.strip().split()[1:]
                face = []
                for p in parts:
                    # supports: "i", "i/uv", "i/uv/n", "i//n"
                    idx = p.split("/")[0]
                    if idx:
                        face.append(int(idx) - 1)  # OBJ is 1-based
                if len(face) >= 3:
                    mesh_f.append(face)

    return np.array(mesh_vectors, dtype=float), mesh_f


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


def load_image(image_path=IMAGE_DIR + "/0000.png"):
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
            
def load_extrinsics_by_name(image_name, extrinsics_source=IMAGES_TXT_PATH):
    target = os.path.basename(image_name)
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


def _read_available_frame_indices(images_txt_path):
    frames = []
    pat = re.compile(r'(\d{4})\.png$', re.IGNORECASE)
    with open(images_txt_path, "r") as f:
        for line in f:
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            for token in line.strip().split():
                m = pat.search(os.path.basename(token))
                if m:
                    frames.append(int(m.group(1)))
                    break
    frames = sorted(set(frames))
    return frames


def render_sequence():
    frames = _read_available_frame_indices(IMAGES_TXT_PATH)
    if not frames:
        raise RuntimeError("No frames found in images.txt")    

    for i in range(NUM_FRAMES):
        image_name_for_extrinsics = "0000.png"
        extrinsics = load_extrinsics_by_name(image_name_for_extrinsics, IMAGES_TXT_PATH)
        print("Extrinsics:", extrinsics)

        intrinsics = load_cam_intrinsics(cameras_txt_path, camera_id=extrinsics["camera_id"])
        print("Intrinsics:", intrinsics)

        img_info = load_image(os.path.join(IMAGE_DIR, f"{i:04d}.png"))
        print("Image:", img_info)

        verts, faces = parse_mesh_obj(os.path.join(OBJ_DIR, f"{i:04d}.obj"))
        print("Mesh:", verts.shape, "verts,", len(faces), "faces")

        Xc, Zc = world_to_camera(verts, extrinsics)
        uv_list = project_pinhole(Xc, intrinsics)

        image = img_info["image"]
        H, W = image.shape[:2]

        uv = np.asarray(uv_list, dtype=float)
        uv_int = np.round(uv).astype(int)

        for face in faces:
            for j in range(len(face)):
                a = face[j]
                b = face[(j + 1) % len(face)]

                if a < 0 or b < 0 or a >= len(uv_int) or b >= len(uv_int):
                    continue

                if not (np.isfinite(uv[a]).all() and np.isfinite(uv[b]).all()):
                    continue

                if Zc[a] <= 0 or Zc[b] <= 0:
                    continue

                p1 = uv_int[a]
                p2 = uv_int[b]

                if (p1[0] < 0 or p1[0] >= W or p1[1] < 0 or p1[1] >= H):
                    continue
                if (p2[0] < 0 or p2[0] >= W or p2[1] < 0 or p2[1] >= H):
                    continue

                cv2.line(image, tuple(p1), tuple(p2), (0, 255, 0), 1, lineType=cv2.LINE_AA)

        cv2.imwrite(f"frame_{i:04d}.png", image)


def video_generation():
    
    cap = cv2.VideoCapture(FRAME_GLOB_PATTERN)
    
    if not cap.isOpened():
        print("Error: Could not open image sequence.")
        return
    
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read the first frame.")
        return
    
    frame_height, frame_width = frame.shape[:2]
    
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(OUTPUT_VIDEO_PATH, fourcc, 30.0, (frame_width, frame_height))
    
    out.write(frame)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("End of sequence.")
            break
        out.write(frame)
    
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"Video saved as {OUTPUT_VIDEO_PATH}")
    
    
def main():
    render_sequence()
    video_generation()
    

if __name__ == "__main__":
    main()
