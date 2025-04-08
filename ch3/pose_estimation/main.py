import model
import torch

if __name__ == "__main__":
    # image input
    input_path = "COCO-128-2/..."
    output_path = "result.png"

    # video input
    input_path = "COCO-128-2/..."
    output_path = "result.mp4"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cfg_path = "configs/yoloxpose_m_8xb32-300e_coco-640.py.py"
    ckpt_path = "weights/yoloxpose_m_8xb32-300e_coco-640-84e9a538_20230829.pth"

    estimator = model.initialize(cfg_path, ckpt_path, device)

    # image result
    results = model.inference(input_path, estimator)
    model.visualize(input_path, output_path, estimator, results)

    # video result
    results = model.inference_and_save_video(input_path, output_path, estimator)
