import os

import cv2
import numpy as np
import torch
from sam2.build_sam import build_sam2_video_predictor
from ultralytics import YOLO


def get_bbox_prompt(frames_dir, dectection_ckpt):
    first_frame_input = frames_dir + "/1.jpg"

    detection_model = YOLO(detection_ckpt)
    detection_results = detection_model(input_img)[0]

    bboxes = detection_results.boxes.xyxy
    bboxes_cpu = bboxes.to("cpu").numpy()
    # print(bboxes)

    prompts = {}

    for idx, bbox in enumerate(bboxes_cpu):
        prompts[idx] = (int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]), 0)

    return prompts


if __name__ == "__main__":
    detection_ckpt = "../yolo11s.pt"
    samurai_cfg = "./sam2/sam2/configs/sam2.1/sam2.1_hiera_s.yaml"
    samurai_ckpt = "./sam2/checkpoints/sam2.1_hiera_small.pt"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    input_img = "data/frame/anything.png"
    frames_dir = "data/frames_short"
    output_path = "result.mp4"

    samurai_predictor = build_sam2_video_predictor(samurai_cfg, samurai_ckpt, device=device)

    prompts = get_bbox_prompt(frames_dir, detection_ckpt)

    frames = sorted([os.path.join(frames_dir, f) for f in os.listdir(frames_dir)])
    loaded_frames = [cv2.imread(frame_path) for frame_path in frames]
    height, width = loaded_frames[0].shape[:2]
    frame_rate = 30

    fourcc = cv2.VideoWriter_fourcc("mp4v")
    output_video = cv2.VideoWriter(output_path, fourcc, frame_rate, (width, height))

    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
        initial_state = samurai_predictor.init_state(frames_dir, offload_video_to_cpu=True)
        bbox, tracking_label = prompts[0]

        _, _, masks = samurai_predictor.add_new_points_or_box(initial_state, box=bbox, frame_idx=0, obj_id=0)

        for frame_idx, object_ids, mask in samurai_predictor.propagate_in_video(initial_state):
            mask_visualize = {}
            bbox_visualize = {}

            for object_id, mask in zip(object_ids, masks):
                mask = mask[0].cpu().numpy()
                mask = mask > 0.0
                non_zero
