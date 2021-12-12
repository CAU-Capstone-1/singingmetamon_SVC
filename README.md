# singingmetamon_SVC
캡스톤 디자인(1) 한국어 가창 음성 변환 시스템 by Team 노래하는 메타몽 

## Prerequisites
- NVIDIA GPU + CUDA CuDNN
- Anaconda

## Installation
- 해당 레포지토리 clone
```bash
git clone https://github.com/2JooYeon/CycleGAN-BEGAN-SVC.git
```
- 아나콘다 가상환경 생성
```bash
conda create -n (env_name) tensorflow-gpu
```
- tensorflow-addons 설치
```bash
pip install tensorflow-addons
```
- librosa 설치
```bash
pip install librosa
```
- pyworld 설치
```bash
pip install pyworld
```

## Prepare Data
- 음성 변환을 진행할 source 화자와 target 화자의 음원이 분리된 음성 데이터를 준비합니다.
- 이후 음성 데이터의 폴더 경로와 전처리 후 저장할 폴더 경로를 각각 audio_path, save_path 변수에 저장합니다.
- 차례로 파일을 실행하여 전처리를 진행합니다.

```
python time_cutting&trim.py 
python augmentation.py 
```





## Extract Voice Feature
- 음성 변환을 진행할 화자 A와 화자 B의 음성 feature를 추출합니다.
```bash
python extracting_feature.py --train_A_dir (준비한 훈련 데이터A 폴더 경로) 
                             --train_B_dir (준비한 훈련 데이터B 폴더 경로) 
                             --cache_folder (추출된 음성 feature 저장할 폴더 경로)
```

## Train Model
- 추출한 음성 feature를 사용해서 스펙트럼 포락선 변환 학습을 진행합니다.
```bash
python train.py --logf0s_normalization (추출된 음성 feature 저장된 폴더 경로)/logf0s_normalization.npz 
                --mcep_normalization (추출된 음성 feature 저장된 폴더 경로)/mcep_normalization.npz 
                --coded_sps_A_norm (추출된 음성 feature 저장된 폴더 경로)/coded_sps_A_norm.pickle 
                --coded_sps_B_norm (추출된 음성 feature 저장된 폴더 경로/coded_sps_B_norm.pickle 
                --model_checkpoint_dir (체크포인트 모델 저장할 폴더 경로)
                --model_checkpoint_file (저장할 체크포인트 이름) 
                --log_dir (로그 파일 저장할 경로) 
                --cuda (사용할 GPU Device 번호)
                --tensorboard_dir (tensorboard 저장할 경로) 
                --resume_training_at (이어서 훈련을 진행할 체크포인터 파일이 들어있는 폴더 경로)
```

## Test Model
- 추출된 음성 feature와 훈련된 모델을 사용해서 음성 변환 및 음성 합성을 진행합니다.
```bash
python test.py --logf0s_normalization (추출된 음성 feature 저장된 폴더 경로)/logf0s.npz 
               --mcep_normalization (추출된 음성 feature 저장된 폴더 경로)/mcep.npz 
               --model_checkpoint_dir (테스트를 진행할 모델이 저장된 폴더 경로)
               --validation_A_dir (준비한 테스트 데이터A 폴더 경로)
               --validation_B_dir (준비한 테스트 데이터B 폴더 경로) 
               --cuda (사용할 GPU Device 번호) 
               --output_A_dir (A의 데이터에서 음성만 B로 바뀐 결과를 저장할 폴더 경로)
               --output_B_dir (B의 데이터에서 음성만 A로 바뀐 결과를 저장할 폴더 경로)
```
