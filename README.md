# !After Detailer

!After Detailer is a extension for stable diffusion webui, similar to Detection Detailer, except it uses ultralytics instead of the mmdet.

## Model

| Model                | Target              | mAP 50 | mAP 50-95 |
| -------------------- | ------------------- | ------ | --------- |
| face_yolov8n.pt      | 2D / realistic face | 0.660  | 0.366     |
| face_yolov8s.pt      | 2D / realistic face | 0.713  | 0.404     |
| mediapipe_face_full  | realistic           | -      | -         |
| mediapipe_face_short | realistic           | -      | -         |

The yolo models can be found on huggingface [Bingsu/adetailer](https://huggingface.co/Bingsu/adetailer).

### Dataset

Datasets used for training the yolo face detection models are:

- [roboflow AN](https://universe.roboflow.com/sed-b8vkf/an-lfg5i)
- [roboflow Anime Face CreateML](https://universe.roboflow.com/my-workspace-mph8o/anime-face-createml)
- [roboflow xml2txt](https://universe.roboflow.com/0oooooo0/xml2txt-njqx1)
- [wider face](http://shuoyang1213.me/WIDERFACE/index.html)

### User Model

Put your [ultralytics](https://github.com/ultralytics/ultralytics) model in `webui/models/adetailer`. The model name should end with `.pt` or `.pth`.

It must be a bbox detection model and use only label 0.

## ControlNet Inpainting

You can use the ControlNet inpaint extension if you have ControlNet installed and a ControlNet inpaint model.

On the ControlNet tab, select a ControlNet inpaint model and set the model weights.

## Example

![image](https://i.imgur.com/i74ukgi.png)
![image](https://i.imgur.com/I5VVkoh.png)
