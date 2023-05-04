# !After Detailer

!After Detailer is a extension for stable diffusion webui, similar to Detection Detailer, except it uses ultralytics instead of the mmdet.

## Install

(from Mikubill/sd-webui-controlnet)

1. Open "Extensions" tab.
2. Open "Install from URL" tab in the tab.
3. Enter `https://github.com/Bing-su/adetailer.git` to "URL for extension's git repository".
4. Press "Install" button.
5. Wait 5 seconds, and you will see the message "Installed into stable-diffusion-webui\extensions\adetailer. Use Installed tab to restart".
6. Go to "Installed" tab, click "Check for updates", and then click "Apply and restart UI". (The next time you can also use this method to update extensions.)
7. Completely restart A1111 webui including your terminal. (If you do not know what is a "terminal", you can reboot your computer: turn your computer off and turn it on again.)

You **DON'T** need to download any model from huggingface.

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
