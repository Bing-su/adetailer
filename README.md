# !After Detailer

!After Detailer is a extension for stable diffusion webui, similar to Detection Detailer, except it uses ultralytics instead of the mmdet.

## Model

| Model                | Target         | mAP 50-95 |
| -------------------- | -------------- | --------- |
| face_yolov8n.pt      | 2D / realistic | 0.366     |
| face_yolov8s.pt      | 2D / realistic | 0.404     |
| mediapipe_face_full  | realistic      | -         |
| mediapipe_face_short | realistic      | -         |

The yolo models can be found on huggingface [Bingsu/adetailer](https://huggingface.co/Bingsu/adetailer).

### Dataset

Datasets used for training the yolo models are:

- [roboflow AN](https://universe.roboflow.com/sed-b8vkf/an-lfg5i)
- [roboflow Anime Face CreateML](https://universe.roboflow.com/my-workspace-mph8o/anime-face-createml)
- [roboflow xml2txt](https://universe.roboflow.com/0oooooo0/xml2txt-njqx1)
- [wider face](http://shuoyang1213.me/WIDERFACE/index.html)

## Example

![image](https://i.imgur.com/i74ukgi.png)
![image](https://i.imgur.com/I5VVkoh.png)
