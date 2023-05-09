# Changelog

### 2023-05-09

- v23.5.10
- 선택한 스크립트만 ADetailer에 적용하는 옵션 추가, 기본값 `True`. 설정 탭에서 지정가능.
  - 기본값: `dynamic_prompting,dynamic_thresholding,wildcards,wildcard_recursive`
- `person_yolov8s-seg.pt` 모델 추가

### 2023-05-08

- v23.5.9
- 2가지 이상의 모델을 사용할 수 있음. 기본값: 2, 최대: 5
- segment 모델을 사용할 수 있게 함. `person_yolov8n-seg.pt` 추가

### 2023-05-07

- v23.5.8
- 프롬프트와 네거티브 프롬프트에 방향키 지원 (PR #24)
- `mask_preprocess`를 추가함. 이전 버전과 시드값이 달라질 가능성 있음!
- 이미지 처리가 일어났을 때에만 before이미지를 저장함
- 설정창의 레이블을 ADetailer 대신 더 적절하게 수정함

### 2023-05-06

- v23.5.7
- `ad_use_cfg_scale` 옵션 추가. cfg 스케일을 따로 사용할지 말지 결정함.
- `ad_enable` 기본값을 `True`에서 `False`로 변경
- `ad_model`의 기본값을 `None`에서 첫번째 모델로 변경
- 최소 2개의 입력(ad_enable, ad_model)만 들어오면 작동하게 변경.

- v23.5.7.post0
- `init_controlnet_ext`을 controlnet_exists == True일때에만 실행
- webui를 C드라이브 바로 밑에 설치한 사람들에게 `ultralytics` 경고 표시

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
