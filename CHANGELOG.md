# Changelog

## 2024-05-19

- v24.5.0
- 개별 탭 활성화/비활성화 체크박스 추가
- ad_extra_model_dir 옵션에 |로 구분된 여러 디렉토리를 추가할 수 있게 함 (PR #596)
- `hypertile` 빌트인 확장이 지원되도록 함
- 항상 cond 캐시를 비움
- 설치 스크립트에 uv를 사용함
- mediapipe 최소 버전을 올려 protobuf 버전 4를 사용하게 함

## 2024-04-17

- v24.4.2
- `params.txt` 파일이 없을 때 에러가 발생하지 않도록 수정
- 파이썬 3.9 이하에서 유니온 타입 에러 방지

## 2024-04-14

- v24.4.1
- webui 1.9.0에서 발생한 에러 수정
  - extra generation params에 callable이 들어와서 생긴 문제
  - assign_current_image에 None이 들어갈 수 있던 문제
- webui 1.9.0에서 변경된 scheduler 지원
- 컨트롤넷 모델을 찾을 때, 대소문자 구분을 하지 않음 (PR #577)
- 몇몇 기능을 스크립트에서 분리하여 별도 파일로 빼냄

## 2024-04-10

- v24.4.0
- txt2img에서 hires를 설정했을 때, 이미지의 exif에서 Denoising Strength가 adetailer의 denoisiog stregnth로 덮어 쓰이는 문제 수정
- ad prompt, ad negative prompt에 프롬프트를 변경하는 기능을 적용했을 때(와일드카드 등), 적용된 프롬프트가 이미지의 exif에 제대로 표시됨

## 2024-03-29

- v24.3.5
- 알 수 없는 이유로 인페인팅을 확인하는 과정에서 Txt2Img 인스턴스가 들어오는 문제에 대한 임시 해결

## 2024-03-28

- v24.3.4
- 인페인트에서, 이미지 해상도가 16의 배수가 아닐 때 사이즈 불일치로 인한 opencv 에러 방지

## 2024-03-25

- v24.3.3
- webui 1.6.0 미만 버전에서 create_binary_mask 함수에 대해 ImportError가 발생하는 것 수정

## 2024-03-21

- v24.3.2
- UI를 거치지 않은 입력에 대해, image_mask를 입력했을 때 opencv 에러가 발생하는 것 수정
- img2img inpaint에서 skip img2img 옵션을 활성화할 경우, adetailer를 비활성화함
  - 마스크 크기에 대해 해결하기 힘든 문제가 있음

## 2024-03-16

- v24.3.1
- YOLO World v2, YOLO9 지원가능한 버전으로 ultralytics 업데이트
- inpaint full res인 경우 인페인트 모드에서 동작하게 변경
- inpaint full res가 아닌 경우, 사용자가 입력한 마스크와 교차점이 있는 마스크만 선택하여 사용함

## 2024-03-01

- v24.3.0
- YOLO World 모델 추가: 가장 큰 yolov8x-world.pt 모델만 기본적으로 선택할 수 있게 함.
- lllyasviel/stable-diffusion-webui-forge에서 컨트롤넷을 사용가능하게 함 (PR #517)
- 기본 스크립트 목록에 soft_inpainting 추가 (https://github.com/AUTOMATIC1111/stable-diffusion-webui/pull/14208)

  - 기존에 설치한 사람에게 소급적용되지는 않음

- 감지모델에 대한 간단한 pytest 추가함
- xyz grid 컨트롤넷 모델 옵션에 `Passthrough` 추가함

## 2024-01-23

- v24.1.2
- controlnet 모델에 `Passthrough` 옵션 추가. 입력으로 들어온 컨트롤넷 옵션을 그대로 사용
- fastapi 엔드포인트 추가

## 2024-01-10

- v24.1.1
- SDNext 호환 업데이트 (issue #466)
  - 설정 값 state에 초기값 추가
  - 위젯 값을 변경할 때마다 state도 변경되게 함 (기존에는 생성 버튼을 누를 때 적용되었음)
- `inpaint_depth_hand` 컨트롤넷 모델이 depth 모델로 인식되게 함 (issue #463)

## 2024-01-04

- v24.1.0
- `depth_hand_refiner` ControlNet 추가 (PR #460)

## 2023-12-30

- v23.12.0
- 파일을 인자로 추가하는 몇몇 스크립트에 대해 deepcopy의 에러를 피하기 위해 script_args 복사 방법을 변경함
- skip img2img 기능을 사용할 때 너비, 높이를 128로 고정하여 스킵 과정이 조금 더 나아짐
- img2img inpainting 모드에서 adetailer 자동 비활성화
- 처음 생성된 params.txt 파일을 항상 유지하도록 변경함

## 2023-11-19

- v23.11.1
- 기본 스크립트 목록에 negpip 추가
  - 기존에 설치한 사람에게 소급적용되지는 않음
- skip img2img 옵션이 2스텝 이상일 때, 제대로 적용되지 않는 문제 수정
- SD.Next에서 이미지가 np.ndarray로 입력되는 경우 수정
- 컨트롤넷 경로를 sys.path에 추가하여 --data-dir등을 지정한 경우에도 임포트 에러가 일어나지 않게 함.

## 2023-10-30

- v23.11.0
- 이미지의 인덱스 계산방법 변경
  - webui 1.1.0 미만에서 adetailer 실행 불가능하게 함
- 컨트롤넷 preprocessor 선택지 늘림
- 추가 yolo 모델 디렉터리를 설정할 수 있는 옵션 추가
- infotext에 `/`가 있는 항목이 exif에서 복원되지 않는 문제 수정
  - 이전 버전에 생성된 이미지는 여전히 복원안됨
- 같은 탭에서 항상 같은 시드를 적용하게 하는 옵션 추가
- 컨트롤넷 1.1.411 (f2aafcf2beb99a03cbdf7db73852228ccd6bd1d6) 버전을 사용중일 경우,
  webui 버전 1.6.0 미만에서 사용할 수 없다는 메세지 출력

## 2023-10-15

- v23.10.1
- xyz grid에 prompt S/R 추가
- img2img에서 steps가 1일때 에러가 발생하는 샘플러의 처리를 위해 샘플러 이름도 변경하게 수정

## 2023-10-07

- v23.10.0
- 허깅페이스 모델을 다운로드 실패했을 때, 계속 다운로드를 시도하지 않음
- img2img에서 img2img단계를 건너뛰는 기능 추가
- live preview에서 감지 단계를 보여줌 (PR #352)

## 2023-09-20

- v23.9.3
- ultralytics 버전 8.0.181로 업데이트 (https://github.com/ultralytics/ultralytics/pull/4891)
- mediapipe와 ultralytics의 lazy import

## 2023-09-10

- v23.9.2
- (실험적) VAE 선택 기능

## 2023-09-01

- v23.9.1
- webui 1.6.0에 추가된 인자를 사용해서 생긴 하위 호환 문제 수정

## 2023-08-31

- v23.9.0
- (실험적) 체크포인트 선택기능
  - 버그가 있어 리프레시 버튼은 구현에서 빠짐
- 1.6.0 업데이트에 따라 img2img에서 사용불가능한 샘플러를 선택했을 때 더이상 Euler로 변경하지 않음
- 유효하지 않은 인자가 전달되었을 때, 에러를 일으키지 않고 대신 adetailer를 비활성화함

## 2023-08-25

- v23.8.1
- xyz grid에서 model을 `None`으로 설정한 이후에 adetailer가 비활성화 되는 문제 수정
- skip을 눌렀을 때 진행을 멈춤
- `--medvram-sdxl`을 설정했을 때에도 cpu를 사용하게 함

## 2023-08-14

- v23.8.0
- `[PROMPT]` 키워드 추가. `ad_prompt` 또는 `ad_negative_prompt`에 사용하면 입력 프롬프트로 대체됨 (PR #243)
- Only top k largest 옵션 추가 (PR #264)
- ultralytics 버전 업데이트

## 2023-07-31

- v23.7.11
- separate clip skip 옵션 추가
- install requirements 정리 (ultralytics 새 버전, mediapipe~=3.20)

## 2023-07-28

- v23.7.10
- ultralytics, mediapipe import문 정리
- traceback에서 컬러를 없앰 (api 때문), 라이브러리 버전도 보여주게 설정.
- huggingface_hub, pydantic을 install.py에서 없앰
- 안쓰는 컨트롤넷 관련 코드 삭제

## 2023-07-23

- v23.7.9
- `ultralytics.utils` ModuleNotFoundError 해결 (https://github.com/ultralytics/ultralytics/issues/3856)
- `pydantic` 2.0 이상 버전 설치안되도록 함
- `controlnet_dir` cmd args 문제 수정 (PR #107)

## 2023-07-20

- v23.7.8
- `paste_field_names` 추가했던 것을 되돌림

## 2023-07-19

- v23.7.7
- 인페인팅 단계에서 별도의 샘플러를 선택할 수 있게 옵션을 추가함 (xyz그리드에도 추가)
- webui 1.0.0-pre 이하 버전에서 batch index 문제 수정
- 스크립트에 `paste_field_names`을 추가함. 사용되는지는 모르겠음

## 2023-07-16

- v23.7.6
- `ultralytics 8.0.135`에 추가된 cpuinfo 기능을 위해 `py-cpuinfo`를 미리 설치하게 함. (미리 설치 안하면 cpu나 mps사용할 때 재시작해야함)
- init_image가 RGB 모드가 아닐 때 RGB로 변경.

## 2023-07-07

- v23.7.4
- batch count > 1일때 프롬프트의 인덱스 문제 수정

- v23.7.5
- i2i의 `cached_uc`와 `cached_c`가 p의 `cached_uc`와 `cached_c`가 다른 인스턴스가 되도록 수정

## 2023-07-05

- v23.7.3
- 버그 수정
  - `object()`가 json 직렬화 안되는 문제
  - `process`를 호출함에 따라 배치 카운트가 2이상일 때, all_prompts가 고정되는 문제
  - `ad-before`와 `ad-preview` 이미지 파일명이 실제 파일명과 다른 문제
  - pydantic 2.0 호환성 문제

## 2023-07-04

- v23.7.2
- `mediapipe_face_mesh_eyes_only` 모델 추가: `mediapipe_face_mesh`로 감지한 뒤 눈만 사용함.
- 매 배치 시작 전에 `scripts.postprocess`를, 후에 `scripts.process`를 호출함.
  - 컨트롤넷을 사용하면 소요 시간이 조금 늘어나지만 몇몇 문제 해결에 도움이 됨.
- `lora_block_weight`를 스크립트 화이트리스트에 추가함.
  - 한번이라도 ADetailer를 사용한 사람은 수동으로 추가해야함.

## 2023-07-03

- v23.7.1
- `process_images`를 진행한 뒤 `StableDiffusionProcessing` 오브젝트의 close를 호출함
- api 호출로 사용했는지 확인하는 속성 추가
- `NansException`이 발생했을 때 중지하지 않고 남은 과정 계속 진행함

## 2023-07-02

- v23.7.0
- `NansException`이 발생하면 로그에 표시하고 원본 이미지를 반환하게 설정
- `rich`를 사용한 에러 트레이싱
  - install.py에 `rich` 추가
- 생성 중에 컴포넌트의 값을 변경하면 args의 값도 함께 변경되는 문제 수정 (issue #180)
- 터미널 로그로 ad_prompt와 ad_negative_prompt에 적용된 실제 프롬프트 확인할 수 있음 (입력과 다를 경우에만)

## 2023-06-28

- v23.6.4
- 최대 모델 수 5 -> 10개
- ad_prompt와 ad_negative_prompt에 빈칸으로 놔두면 입력 프롬프트가 사용된다는 문구 추가
- huggingface 모델 다운로드 실패시 로깅
- 1st 모델이 `None`일 경우 나머지 입력을 무시하던 문제 수정
- `--use-cpu` 에 `adetailer` 입력 시 cpu로 yolo모델을 사용함

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
