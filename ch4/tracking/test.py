import torch
from ultralytics import YOLO

if __name__ == "__main__":
    detection_ckpt = "../yolo11s.pt"
    samurai_cfg = "./sam2/sam2/configs/sam2.1/sam2.1_hiera_s.yaml"
    samurai_ckpt = "./sam2/checkpoints/sam2.1_hiera_small.pt"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    input_img = "data/frame/anything.png"
    frames_dir = "data/frames_short"
    output_path = "result.mp4"

    detection_model = YOLO(detection_ckpt)
    detection_results = detection_model(input_img)[0]

    bboxes = detection_results.boxes.xyxy
    bboxes_cpu = bboxes.to("cpu").numpy()
    # print(bboxes)

    prompts = {}

    for idx, bbox in enumerate(bboxes_cpu):
        prompts[idx] = (bbox, 0)
