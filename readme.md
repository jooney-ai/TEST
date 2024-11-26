# CRAFT-train
공식 CRAFT github에는 CRAFT 모델을 훈련하고자 하는 많은 사람들이 있습니다.

하지만 훈련 코드는 공식 CRAFT 저장소에 게시되지 않았습니다.

다른 재생산된 코드도 있지만, 그 성능과 원래 논문에 보고된 성능 사이에 차이가 있습니다. ( https://arxiv.org/pdf/1904.01941.pdf )

이 코드를 사용하여 훈련된 모델은 원래 논문과 유사한 수준의 성능을 기록했습니다.



```bash
├── config
│   ├── syn_train.yaml
│   └── custom_data_train.yaml
├── data
│   ├── pseudo_label
│   │   ├── make_charbox.py
│   │   └── watershed.py
│   ├── boxEnlarge.py
│   ├── dataset.py
│   ├── gaussian.py
│   ├── imgaug.py
│   └── imgproc.py
├── loss
│   └── mseloss.py
├── metrics
│   └── eval_det_iou.py
├── model
│   ├── craft.py
│   └── vgg16_bn.py
├── utils
│   ├── craft_utils.py
│   ├── inference_boxes.py
│   └── utils.py
├── trainSynth.py
├── train.py
├── train_distributed.py
├── eval.py
├── data_root_dir   (place dataset folder here)
└── exp             (model and experiment result files will saved here)
```

### Installation

`pip`를 사용하여 설치pip

``` bash
pip install -r requirements.txt
```


### Training
1. 다음 형식으로 훈련 및 데이터를 배치하세요.
    ```
    └── data_root_dir (you can change root dir in yaml file)
        ├── ch4_training_images
        │   ├── img_1.jpg
        │   └── img_2.jpg
        ├── ch4_training_localization_transcription_gt
        │   ├── gt_img_1.txt
        │   └── gt_img_2.txt
        ├── ch4_test_images
        │   ├── img_1.jpg
        │   └── img_2.jpg
        └── ch4_training_localization_transcription_gt
            ├── gt_img_1.txt
            └── gt_img_2.txt
    ```
   * localization_transcription_gt 파일의 형식 :
   ```
    377,117,463,117,465,130,378,130,Genaxis Theatre
    493,115,519,115,519,131,493,131,[06]
    374,155,409,155,409,170,374,170,###
    ```
2. YAML 형식으로 구성을 작성합니다. (예시 구성 파일은 `config` 폴더에 제공됩니다.)
    * 다중 GPU로 학습 시간을 단축하려면 num_worker > 0으로 설정하세요.
3. yaml 파일을 config 폴더에 넣으세요
4. 아래와 같이 훈련 스크립트를 실행합니다. (다중 GPU가 있는 경우 train_distributed.py를 실행합니다.)
5. 그러면 실험 결과가 기본적 으로 ```./exp/[yaml]``` 에 저장됩니다 .

* 1단계: SynthText 데이터세트를 사용하여 처음부터 CRAFT를 훈련합니다.
    * 참고 : 2단계 학습을 시작할 때  <a href="https://drive.google.com/file/d/1enVIsgNvBf3YiRsVkxodspOn55PIK-LJ/view?usp=sharing">이 pretrain을</a> 체크포인트로 사용하는 경우 이 단계는 필요하지 않습니다. 다운로드하여 로컬 설정에 따라 구성 파일에 `exp/CRAFT_clr_amp_29500.pth` 를 넣고 `ckpt_path`를 변경하여 진행할 수 있습니다. 
    ```
    CUDA_VISIBLE_DEVICES=0 python3 trainSynth.py --yaml=syn_train
    ```

* 2단계 : [SynthText + IC15] 또는 사용자 정의 데이터 세트로 CRAFT를 훈련합니다. 
    ```
    CUDA_VISIBLE_DEVICES=0 python3 train.py --yaml=custom_data_train               ## if you run on single GPU
    CUDA_VISIBLE_DEVICES=0,1 python3 train_distributed.py --yaml=custom_data_train   ## if you run on multi GPU
    ```

### Arguments
* ```--yaml``` : configuration 파일 이름

### Evaluation
* 공식 저장소 문제에서 저자는 첫 번째 행 설정 F1-점수가 약 0.75라고 언급했습니다.
* 공식 문서에서는 두 번째 행 설정의 결과 F1-점수가 0.87이라고 명시되어 있습니다.
    * 사후 처리 매개변수 'text_threshold'를 0.85에서 0.75로 조정하면 F1 점수는 0.856에 도달합니다.
* 8개의 RTX 3090 Ti를 사용하여 약한 감독 25,000 반복을 훈련하는 데 14시간이 걸렸습니다
    * GPU의 절반은 훈련에 할당되고, 나머지 절반은 감독 설정에 할당됩니다.

| Training Dataset   | Evaluation Dataset   | Precision  | Recall  | F1-score  | pretrained model  |
| ------------- |-----|:-----:|:-----:|:-----:|-----:|
| SynthText      |  ICDAR2013 | 0.801 | 0.748 | 0.773| <a href="https://drive.google.com/file/d/1enVIsgNvBf3YiRsVkxodspOn55PIK-LJ/view?usp=sharing">download link</a>|
| SynthText + ICDAR2015      | ICDAR2015  | 0.909 | 0.794 | 0.848| <a href="https://drive.google.com/file/d/1qUeZIDSFCOuGS9yo8o0fi-zYHLEW6lBP/view">download link</a>|


### PNG를 JPG로 변환하기
훈련 파이프라인은 .jpg 파일을 필요로 하므로, 모든 이미지 파일을 .jpg 형식으로 변환해야 합니다. 
아래 스크립트를 실행하세요:
`python png_to_jpg.py`

* 기능: 
    * 지정된 폴더 내의 모든 .png 파일을 .jpg로 변환합니다. 
    * 변환 후 원본 .png 파일은 삭제됩니다. 
* 스크립트: png_to_jpg.py 파일에서 데이터셋 경로를 수정하여 올바른 폴더를 지정하세요.