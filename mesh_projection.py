import numpy as np
import cv2
import json
import csv
import matplotlib.pyplot as plt
import os
import random
import re 

cameras_txt_path = "data/cameras/cameras.txt"

def parse_mesh_obj(obj_path = "data/skin_color_1/objs/0000.obj"):
    
    mesh_vectors = []
    mesh_f = []
    
    ...

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
    frames = _read_available_frame_indices("data/cameras/images.txt")
    if not frames:
        raise RuntimeError("No frames found in images.txt")    

    for i in range(57):
        image_name_for_extrinsics = "0000.png"
        extrinsics = load_extrinsics_by_name(image_name_for_extrinsics, "data/cameras/images.txt")
        print("Extrinsics:", extrinsics)

        intrinsics = load_cam_intrinsics(cameras_txt_path, camera_id=extrinsics["camera_id"])
        print("Intrinsics:", intrinsics)

        img_info = load_image(f"data/skin_color_1/cam_0000/{i:04d}.png")
        print("Image:", img_info)

        
def video_generation():
    
    cap = cv2.VideoCapture("frame_%04d.png")  # use your saved overlay images
    
    if not cap.isOpened():
        print("Error: Could not open image sequence.")
        return
    
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read the first frame.")
        return
    
    frame_height, frame_width = frame.shape[:2]
    
    # Use mp4v for .mp4 output
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter("skeleton.mp4", fourcc, 30.0, (frame_width, frame_height))
    
    # Write first frame
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
    print("Video saved as skeleton.mp4")
    
    
def main():
    render_sequence()
    video_generation()
    

if __name__ == "__main__":
    main()
