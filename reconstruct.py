# reconstruct.py

import os
import argparse
import torch

from mapanything.models import MapAnything
from mapanything.utils.image import preprocess_inputs

from utilities import (
    load_views,
    build_views_from_predictions,
    export_pointcloud_obj,
)


# =========================================================
# ---------------- Reconstruction Pipeline ----------------
# =========================================================

def run_inference(model, views, args):

    processed = preprocess_inputs(views)

    with torch.no_grad():
        predictions = model.infer(
            processed,
            memory_efficient_inference=False,
            use_amp=True,
            amp_dtype="bf16",
            apply_mask=True,
            mask_edges=True,
            apply_confidence_mask=args.conf_mask,
            confidence_percentile=10,
            ignore_calibration_inputs=not args.with_pose,
            ignore_pose_inputs=not args.with_pose,
            ignore_depth_inputs=True,
        )

    return predictions


def reconstruct_scene(args):

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Loading model...")
    model = MapAnything.from_pretrained(
        "facebook/map-anything"
    ).to(device)
    model.eval()

    print("Loading views...")

    if args.with_mask:
        views, masks = load_views(
            args.data_root,
            use_pose=args.with_pose,
            use_mask=True
        )
    else:
        views = load_views(
            args.data_root,
            use_pose=args.with_pose,
            use_mask=False
        )
        masks = None

    print("Running first pass...")
    predictions = run_inference(model, views, args)

    export_pointcloud_obj(
        predictions,
        views,
        args.output + "_first.obj",
        input_masks=masks
    )

    if args.second_pass:

        print("Running second pass...")

        new_views = build_views_from_predictions(
            views, predictions, device
        )

        args.with_pose = True

        predictions_2 = run_inference(model, new_views, args)

        export_pointcloud_obj(
            predictions_2,
            new_views,
            args.output + "_second.obj",
            input_masks=masks
        )
        
        new_views = build_views_from_predictions(
            views, predictions_2, device
        )

        predictions_3 = run_inference(model, new_views, args)

        export_pointcloud_obj(
            predictions_3,
            new_views,
            args.output + "_third.obj",
            input_masks=masks
        )
# =========================================================
# ---------------- CLI Arguments --------------------------
# =========================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--output", type=str, default="reconstruction")
    parser.add_argument("--with_pose", action="store_true")
    parser.add_argument("--second_pass", action="store_true")
    parser.add_argument("--conf_mask", action="store_true")
    parser.add_argument("--with_mask", action="store_true")

    args = parser.parse_args()

    reconstruct_scene(args)
