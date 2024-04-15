import glob
import os
from urllib.parse import urlparse

from huggingface_hub import hf_hub_download

from modules.paths import data_path

urls_huggingface = {
    "face_yolov8m.pt": "https://huggingface.co/Bingsu/adetailer/resolve/main/face_yolov8m.pt",
    "face_yolov8n.pt": "https://huggingface.co/Bingsu/adetailer/resolve/main/face_yolov8n.pt",
    "face_yolov8n_v2.pt": "https://huggingface.co/Bingsu/adetailer/resolve/main/face_yolov8n_v2.pt",
    "face_yolov8s.pt": "https://huggingface.co/Bingsu/adetailer/resolve/main/face_yolov8s.pt",
    "face_yolov9c.pt": "https://huggingface.co/Bingsu/adetailer/resolve/main/face_yolov9c.pt",
    "hand_yolov8n.pt": "https://huggingface.co/Bingsu/adetailer/resolve/main/hand_yolov8n.pt",
    "hand_yolov8s.pt": "https://huggingface.co/Bingsu/adetailer/resolve/main/hand_yolov8s.pt",
    "hand_yolov9c.pt": "https://huggingface.co/Bingsu/adetailer/resolve/main/hand_yolov9c.pt",
    "person_yolov8m-seg.pt": "https://huggingface.co/Bingsu/adetailer/resolve/main/person_yolov8m-seg.pt",
    "person_yolov8n-seg.pt": "https://huggingface.co/Bingsu/adetailer/resolve/main/person_yolov8n-seg.pt",
    "person_yolov8s-seg.pt": "https://huggingface.co/Bingsu/adetailer/resolve/main/person_yolov8s-seg.pt",
    "deepfashion2_yolov8s-seg.pt": "https://huggingface.co/Bingsu/adetailer/resolve/main/deepfashion2_yolov8s-seg.pt",
    "bmab_face_nm_yolov8n.pt": "https://huggingface.co/portu-sim/bmab/resolve/main/bmab_face_nm_yolov8n.pt",
    "bmab_face_sm_yolov8n.pt": "https://huggingface.co/portu-sim/bmab/resolve/main/bmab_face_sm_yolov8n.pt",
    "bmab_hand_yolov8n.pt": "https://huggingface.co/portu-sim/bmab/resolve/main/bmab_hand_yolov8n.pt",
}


def download_models(urls_huggingface):
    # Set the destination folder for Hugging Face models
    adetailer_model_path = os.path.join(data_path, "models/adetailer")
    os.makedirs(adetailer_model_path, exist_ok=True)

    for filename, url in urls_huggingface.items():
        # Extracting repo_id from the URL
        repo_id = url.split("/")[3] + "/" + url.split("/")[4]

        files = glob.glob(os.path.join(adetailer_model_path, "*"))

        if filename not in [os.path.basename(f) for f in files]:
            # Downloading the file with destination folder specified
            hf_hub_download(
                repo_id=repo_id, filename=filename, local_dir=adetailer_model_path
            )
            print(f"Downloaded {filename} to {adetailer_model_path}")


# Call the function to start downloading
# download_models(urls_huggingface)
