import os
import glob
import numpy as np
import cv2
import torch
import trimesh
from pathlib import Path

# =========================================================
# ---------------- Image + Camera Loading -----------------
# =========================================================


def load_mask(mask_path):
    if not os.path.exists(mask_path):
        raise FileNotFoundError(f"Mask not found: {mask_path}")

    mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)

    # If RGB → convert to grayscale
    if len(mask.shape) == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

    # Binarize: white > 127
    mask = (mask > 127).astype(np.uint8)

    return mask


def load_image(path):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def load_camera_txt(path):
    with open(path, "r") as f:
        lines = f.readlines()

    pose_3x4 = np.array(
        [list(map(float, lines[i].split())) for i in range(3)],
        dtype=np.float32,
    )

    pose = np.eye(4, dtype=np.float32)
    pose[:3, :4] = pose_3x4

    intrinsics = np.array(
        [list(map(float, lines[i].split())) for i in range(3, 6)],
        dtype=np.float32,
    )

    return pose, intrinsics


def get_image_paths(image_dir):
    extensions = ["*.jpg", "*.JPG", "*.jpeg", "*.JPEG", "*.png", "*.PNG"]
    
    image_paths = []
    for ext in extensions:
        image_paths.extend(glob.glob(os.path.join(image_dir, ext)))

    return sorted(image_paths)

def load_views(root_dir, use_pose=False, use_mask=False):

    image_dir = os.path.join(root_dir, "images")
    mask_dir  = os.path.join(root_dir, "masks")

    image_paths = get_image_paths(image_dir)

    views = []

    masks = []

    for img_path in image_paths:

        name = Path(img_path).name
        img = load_image(img_path)

        view = {"img": img}

        # --------------------------
        # Apply mask if enabled
        # --------------------------
        if use_mask:
            mask_path = os.path.join(mask_dir, f"{name}")

            mask = load_mask(mask_path)

            # Resize mask if needed
            if mask.shape[:2] != img.shape[:2]:
                mask = cv2.resize(
                    mask,
                    (img.shape[1], img.shape[0]),
                    interpolation=cv2.INTER_NEAREST
                )

            # Apply mask to image
            img_masked = img.copy()
            img_masked[mask == 0] = 0

            view["img"] = img_masked
            # view["mask_input"] = mask   # save for later OBJ filtering
            masks.append(mask) 

        # --------------------------
        # Pose if needed
        # --------------------------
        if use_pose:
            pose, K = load_camera_txt(...)
            view["camera_poses"] = torch.tensor(pose)
            view["intrinsics"] = torch.tensor(K)

        views.append(view)

    if use_mask:
        return views, masks
    else:    
        return views


# =========================================================
# ---------------- Prediction Utilities -------------------
# =========================================================

def build_views_from_predictions(views, predictions, device):
    new_views = []

    for view, pred in zip(views, predictions):

        new_views.append(
            {
                "img": view["img"],
                "intrinsics": pred["intrinsics"][0].cpu(),
                "camera_poses": pred["camera_poses"][0].cpu(),
                "is_metric_scale": torch.tensor([True], device=device),
            }
        )

    return new_views


# =========================================================
# ---------------- Export Utilities -----------------------
# =========================================================

def export_pointcloud_obj(predictions, views, path, input_masks=None):

    pts_list, col_list = [], []

    for i, (pred, view) in enumerate(zip(predictions, views)):

        pts = pred["pts3d"][0].cpu().numpy().reshape(-1, 3)
        mask_model = pred["mask"][0].cpu().numpy().reshape(-1)
        img = pred["img_no_norm"][0].cpu().numpy().reshape(-1, 3) * 255 *1.3
        pred_h, pred_w = pred["img_no_norm"][0].shape[:2]

        print(pred["mask"][0].shape, input_masks[0].shape)
        valid = mask_model > 0

        # --------------------------------
        # Apply input mask if available
        # --------------------------------
        if input_masks is not None and input_masks[i] is not None:
            if input_masks[i].shape[:2] != (pred_h, pred_w):
                mask_input = cv2.resize(
                    input_masks[i],
                    (pred_w, pred_h),
                    interpolation=cv2.INTER_NEAREST
                )
            mask_input = mask_input.reshape(-1)
            valid = valid & (mask_input > 0)

        pts_list.append(pts[valid])
        col_list.append(img[valid])

    if len(pts_list) == 0:
        print("[WARNING] No valid points found.")
        return

    pts_all = np.concatenate(pts_list)
    col_all = np.concatenate(col_list) / 255.0

    with open(path, "w") as f:
        for p, c in zip(pts_all, col_all):
            f.write(
                f"v {p[0]} {p[1]} {p[2]} "
                f"{c[0]} {c[1]} {c[2]}\n"
            )

    print(f"[OBJ] Saved: {path}")



# import os
# import cv2

# # ====== Paths ======
# input_folder = "data/Multi_View_teeth/images"
# output_folder = "data/Multi_View_teeth/masks"

# # Create output folder if not exists
# os.makedirs(output_folder, exist_ok=True)

# # Threshold value
# threshold_value = 5  # change if needed

# # ====== Process Images ======
# for filename in os.listdir(input_folder):
#     file_path = os.path.join(input_folder, filename)

#     # Check if file is image
#     if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')):
        
#         # Load color image
#         img = cv2.imread(file_path)

#         if img is None:
#             continue  # skip corrupted images

#         # Convert to grayscale
#         gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#         # Apply threshold
#         _, binary = cv2.threshold(gray, threshold_value, 255, cv2.THRESH_BINARY)

#         # Save to output folder with same name
#         save_path = os.path.join(output_folder, filename)
#         cv2.imwrite(save_path, binary)

# print("Done processing all images.")