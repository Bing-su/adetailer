import platform
from pathlib import Path
from typing import Union

import cv2
from PIL import Image

from adetailer import PredictOutput
from adetailer.common import create_mask_from_bbox

checked = False


def ultralytics_predict(
    model_path: Union[str, Path],
    image: Image.Image,
    confidence: float = 0.3,
    device: str = "",
) -> PredictOutput:
    if not checked:
        ultralytics_check()

    from ultralytics import YOLO

    model_path = str(model_path)

    model = YOLO(model_path)
    pred = model(image, conf=confidence, show_labels=False, device=device)

    bboxes = pred[0].boxes.xyxy.cpu().numpy()
    if bboxes.size == 0:
        return PredictOutput()
    bboxes = bboxes.tolist()

    if pred[0].masks is None:
        masks = create_mask_from_bbox(image, bboxes)
    else:
        masks = mask_to_pil(pred[0].masks.data)
    preview = pred[0].plot()
    preview = cv2.cvtColor(preview, cv2.COLOR_BGR2RGB)
    preview = Image.fromarray(preview)

    return PredictOutput(bboxes=bboxes, masks=masks, preview=preview)


def ultralytics_check():
    global checked

    checked = True
    if platform.system() != "Windows":
        return

    p = str(Path.cwd().parent)
    if p == "C:\\":
        message = "[-] ADetailer: if you get stuck here, try moving the stable-diffusion-webui to a different directory, or try running as administrator."
        print(message)


def mask_to_pil(masks) -> list[Image.Image]:
    """
    Parameters
    ----------
    masks: torch.Tensor, dtype=torch.float32, shape=(N, H, W).
    The device can be CUDA, but `to_pil_image` takes care of that.
    """
    from torchvision.transforms.functional import to_pil_image

    n = masks.shape[0]
    return [to_pil_image(masks[i], mode="L") for i in range(n)]
