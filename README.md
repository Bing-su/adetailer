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

You can now install it directly from the Extensions tab.

![image](https://i.imgur.com/g6GdRBT.png)

You **DON'T** need to download any model from huggingface.

## Usage

It's auto detecting, masking, and inpainting tool.

So some options correspond to options on the inpaint tab.

![image](https://i.imgur.com/Bm7YLEA.png)

Other options:

| Option                                 |                                                                                              |                                                                                         |
| -------------------------------------- | -------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------- |
| ADetailer model                        | Determine what to detect.                                                                    | `None` = disable                                                                        |
| ADetailer prompt, negative prompt      | Prompts and negative prompts to apply                                                        | If left blank, it will use the same as the input.                                       |
| Detection model confidence threshold % | Only objects with a detection model confidence above this threshold are used for inpainting. |                                                                                         |
| Mask erosion (-) / dilation (+)        | Enlarge or reduce the detected mask.                                                         | [opencv example](https://docs.opencv.org/4.7.0/db/df6/tutorial_erosion_dilatation.html) |
| Mask x, y offset                       | Moves the mask horizontally and vertically by pixels.                                        |                                                                                         |                                                                 |

See the [wiki](https://github.com/Bing-su/adetailer/wiki) for more options and other features.

## ControlNet Inpainting

You can use the ControlNet inpaint extension if you have ControlNet installed and a ControlNet inpaint model.

On the ControlNet tab, select a ControlNet inpaint model and set the model weights.

## Model

| Model                 | Target                | mAP 50                        | mAP 50-95                     |
| --------------------- | --------------------- | ----------------------------- | ----------------------------- |
| face_yolov8n.pt       | 2D / realistic face   | 0.660                         | 0.366                         |
| face_yolov8s.pt       | 2D / realistic face   | 0.713                         | 0.404                         |
| mediapipe_face_full   | realistic face        | -                             | -                             |
| mediapipe_face_short  | realistic face        | -                             | -                             |
| hand_yolov8n.pt       | 2D / realistic hand   | 0.767                         | 0.505                         |
| person_yolov8n-seg.pt | 2D / realistic person | 0.782 (bbox)<br/>0.761 (mask) | 0.555 (bbox)<br/>0.460 (mask) |
| person_yolov8s-seg.pt | 2D / realistic person | 0.824 (bbox)<br/>0.809 (mask) | 0.605 (bbox)<br/>0.508 (mask) |

The yolo models can be found on huggingface [Bingsu/adetailer](https://huggingface.co/Bingsu/adetailer).

### User Model

Put your [ultralytics](https://github.com/ultralytics/ultralytics) model in `webui/models/adetailer`. The model name should end with `.pt` or `.pth`.

It must be a bbox detection or segment model and use all label.

### Dataset

Datasets used for training the yolo models are:

#### Face

- [Anime Face CreateML](https://universe.roboflow.com/my-workspace-mph8o/anime-face-createml)
- [xml2txt](https://universe.roboflow.com/0oooooo0/xml2txt-njqx1)
- [AN](https://universe.roboflow.com/sed-b8vkf/an-lfg5i)
- [wider face](http://shuoyang1213.me/WIDERFACE/index.html)

#### Hand

- [AnHDet](https://universe.roboflow.com/1-yshhi/anhdet)
- [hand-detection-fuao9](https://universe.roboflow.com/catwithawand/hand-detection-fuao9)

#### Person

- [coco2017](https://cocodataset.org/#home) (only person)
- [AniSeg](https://github.com/jerryli27/AniSeg)
- [skytnt/anime-segmentation](https://huggingface.co/datasets/skytnt/anime-segmentation)

## Example

![image](https://i.imgur.com/38RSxSO.png)
![image](https://i.imgur.com/2CYgjLx.png)

[![ko-fi](https://ko-fi.com/img/githubbutton_sm.svg)](https://ko-fi.com/F1F1L7V2N)
