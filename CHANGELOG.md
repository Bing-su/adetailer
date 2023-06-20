# Changelog

## 2023-06-20

- v23.6.3
- 컨트롤넷 inpaint 모델에 대해, 3가지 모듈을 사용할 수 있도록 함
- Noise Multiplier 옵션 추가 (PR #149)
- pydantic 최소 버전 1.10.8로 설정 (Issue #146)

## 2023-06-05

- v23.6.2
- xyz_grid에서 ADetailer를 사용할 수 있게함.
  - 8가지 옵션만 1st 탭에 적용되도록 함.

## 2023-06-01

- v23.6.1
- `inpaint, scribble, lineart, openpose, tile` 5가지 컨트롤넷 모델 지원 (PR #107)
- controlnet guidance start, end 인자 추가 (PR #107)
- `modules.extensions`를 사용하여 컨트롤넷 확장을 불러오고 경로를 알아내로록 변경
- ui에서 컨트롤넷을 별도 함수로 분리

## 2023-05-30

- v23.6.0
- 스크립트의 이름을 `After Detailer`에서 `ADetailer`로 변경
  - API 사용자는 변경 필요함
- 몇몇 설정 변경
  - `ad_conf` → `ad_confidence`. 0~100 사이의 int → 0.0~1.0 사이의 float
  - `ad_inpaint_full_res` → `ad_inpaint_only_masked`
  - `ad_inpaint_full_res_padding` → `ad_inpaint_only_masked_padding`
- mediapipe face mesh 모델 추가
  - mediapipe 최소 버전 `0.10.0`

- rich traceback 제거함
- huggingface 다운로드 실패할 때 에러가 나지 않게 하고 해당 모델을 제거함

## 2023-05-26

- v23.5.19
- 1번째 탭에도 `None` 옵션을 추가함
- api로 ad controlnet model에 inpaint가 아닌 다른 컨트롤넷 모델을 사용하지 못하도록 막음
- adetailer 진행중에 total tqdm 진행바 업데이트를 멈춤
- state.inturrupted 상태에서 adetailer 과정을 중지함
- 컨트롤넷 process를 각 batch가 끝난 순간에만 호출하도록 변경

### 2023-05-25

- v23.5.18
- 컨트롤넷 관련 수정
  - unit의 `input_mode`를 `SIMPLE`로 모두 변경
  - 컨트롤넷 유넷 훅과 하이잭 함수들을 adetailer를 실행할 때에만 되돌리는 기능 추가
  - adetailer 처리가 끝난 뒤 컨트롤넷 스크립트의 process를 다시 진행함. (batch count 2 이상일때의 문제 해결)
- 기본 활성 스크립트 목록에서 컨트롤넷을 뺌

### 2023-05-22

- v23.5.17
- 컨트롤넷 확장이 있으면 컨트롤넷 스크립트를 활성화함. (컨트롤넷 관련 문제 해결)
- 모든 컴포넌트에 elem_id 설정
- ui에 버전을 표시함


### 2023-05-19

- v23.5.16
- 추가한 옵션
  - Mask min/max ratio
  - Mask merge mode
  - Restore faces after ADetailer
- 옵션들을 Accordion으로 묶음

### 2023-05-18

- v23.5.15
- 필요한 것만 임포트하도록 변경 (vae 로딩 오류 없어짐. 로딩 속도 빨라짐)

### 2023-05-17

- v23.5.14
- `[SKIP]`으로 ad prompt 일부를 건너뛰는 기능 추가
- bbox 정렬 옵션 추가
- sd_webui 타입힌트를 만들어냄
- enable checker와 관련된 api 오류 수정?

### 2023-05-15

- v23.5.13
- `[SEP]`으로 ad prompt를 분리하여 적용하는 기능 추가
- enable checker를 다시 pydantic으로 변경함
- ui 관련 함수를 adetailer.ui 폴더로 분리함
- controlnet을 사용할 때 모든 controlnet unit 비활성화
- adetailer 폴더가 없으면 만들게 함

### 2023-05-13

- v23.5.12
- `ad_enable`을 제외한 입력이 dict타입으로 들어오도록 변경
  - web api로 사용할 때에 특히 사용하기 쉬움
  - web api breaking change
- `mask_preprocess` 인자를 넣지 않았던 오류 수정 (PR #47)
- huggingface에서 모델을 다운로드하지 않는 옵션 추가 `--ad-no-huggingface`

### 2023-05-12

- v23.5.11
- `ultralytics` 알람 제거
- 필요없는 exif 인자 더 제거함
- `use separate steps` 옵션 추가
- ui 배치를 조정함

### 2023-05-09

- v23.5.10
- 선택한 스크립트만 ADetailer에 적용하는 옵션 추가, 기본값 `True`. 설정 탭에서 지정가능.
  - 기본값: `dynamic_prompting,dynamic_thresholding,wildcards,wildcard_recursive`
- `person_yolov8s-seg.pt` 모델 추가
- `ultralytics`의 최소 버전을 `8.0.97`로 설정 (C:\\ 문제 해결된 버전)

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
