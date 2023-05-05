# Changelog

### 2023-05-05 (어린이날)

- v23.5.5
- `Save images before ADetailer` 옵션 추가
- 입력으로 들어온 인자와 ALL_ARGS의 길이가 다르면 에러메세지
- README.md에 설치방법 추가

- v23.5.6
- get_args에서 IndexError가 발생하면 자세한 에러메세지를 볼 수 있음
- AdetailerArgs에 extra_params 내장
- scripts_args를 딥카피함
- postprocess_image를 약간 분리함

- v23.5.6.post0
- `init_controlnet_ext`에서 에러메세지를 자세히 볼 수 있음

### 2023-05-04

- v23.5.4
- use pydantic for arguments validation
- revert: ad_model to `None` as default
- revert: `__future__` imports
- lazily import yolo and mediapipe

### 2023-05-03

- v23.5.3.post0
- remove `__future__` imports
- change to copy scripts and scripts args

- v23.5.3.post1
- change default ad_model from `None`

### 2023-05-02

- v23.5.3
- Remove `None` from model list and add `Enable ADetailer` checkbox.
- install.py `skip_install` fix.
