# OCR README

## OCR 모델 학습 방법

OCR 라이브러리로는 EasyOCR을 사용했으며, OCR 모델 학습을 위해 recognition, detection 두가지 모델을 학습시켜야 합니다.

* **recognition** : 문자 인식 모델
* **detection** : 문자 영역 탐지 모델

## 실행파일

```
root
├── 1.EasyOCR_tutorial.ipynb    # EasyOCR 환경구축 및 모델 학습까지의 과정들이 담겨 있습니다.
├── 2.augmentation.ipynb        # 데이터 증강을 위한 코드가 있습니다.
├── 3.labeling.ipynb            # 데이터셋 구축 시, 레이블링 작업을 위한 코드가 있습니다.
├── 4.recognition_train.ipynb   # recognition 모델 학습 코드가 있습니다.
├── ...
└── craft                       # detection에 필요한 코드들이 담겨 있습니다. 관련 설명은 craft의 README를 참고해주세요.
    ├── train.py
    ├── eval.py
    └── ...
```

recognition의 기반인 deep-text-recognition-benchmark에 대해 자세한 설명은 아래를 참고해주세요.

---

## Scene Text Recognition 모델 비교의 문제점: 데이터셋 및 모델 분석

| [논문 링크](https://arxiv.org/abs/1904.01906) | [훈련 및 평가 데이터 다운로드](https://github.com/clovaai/deep-text-recognition-benchmark#download-lmdb-dataset-for-traininig-and-evaluation-from-here) | [실패 사례 및 정제된 라벨 다운로드](https://github.com/clovaai/deep-text-recognition-benchmark#download-failure-cases-and-cleansed-label-from-here) | [사전 학습된 모델](https://www.dropbox.com/sh/j3xmli4di1zuv3s/AAArdcPgz7UFxIHUuKNOeKv_a?dl=0) | [Baidu 버전(비밀번호: rryk)](https://pan.baidu.com/s/1KSNLv4EY3zFWHpBYlpFCBQ) |

이 코드는 PyTorch를 기반으로 구현된 Scene Text Recognition(STR) 4단계 프레임워크의 공식 버전입니다.
대부분의 기존 STR 모델이 이 프레임워크와 호환됩니다.
이 프레임워크는 일관된 훈련 및 평가 데이터셋을 활용해 정확도, 속도, 메모리 사용량 측면에서 각 모듈의 성능 기여도를 체계적으로 분석할 수 있도록 설계되었습니다.
이를 통해 기존 모듈의 성능 향상을 명확히 이해할 수 있으며, 모델 비교 과정에서 발생하는 혼란을 효과적으로 해소할 수 있습니다.

---

## 업데이트
- **2020년 8월 3일**: Baidu warpctc 사용 가이드 추가, [링크](https://github.com/clovaai/deep-text-recognition-benchmark/pull/209).
- **2019년 12월 27일**: FLOPS 추가 및 기타 업데이트, [링크](https://github.com/clovaai/deep-text-recognition-benchmark/issues/125).
- **2019년 10월 22일**: confidence score 추가, [링크](https://github.com/clovaai/deep-text-recognition-benchmark/issues/82).
- **2019년 7월 31일**: ICCV 2019에서 발표.
- **2019년 7월 16일**: ST 데이터셋에 특수문자 포함 데이터 추가, [링크](https://github.com/clovaai/deep-text-recognition-benchmark/issues/7#issuecomment-511727025).
- **2019년 6월 24일**: 실패 사례 이미지 라벨(gt.txt) 제공, [링크](https://drive.google.com/open?id=1VAP9l5GL5fgptgKDLio_h3nMe7X9W0Mf).
- **2019년 5월 9일**: PyTorch 버전 업데이트(1.0.1 → 1.1.0).

---

## 시작하기

### 요구 사항
- PyTorch 1.3.1, CUDA 10.1, Python 3.6, Ubuntu 16.04에서 테스트되었습니다.
  ```bash
  pip3 install torch==1.3.1

### 환경 설정
이 논문에서는 **PyTorch 0.4.1, CUDA 9.0** 환경에서 실험을 수행했습니다.  
아래 필수 패키지를 설치하세요:
```
pip3 install lmdb pillow torchvision nltk natsort
```

### LMDB 데이터셋 다운로드
훈련 및 평가를 위한 LMDB 데이터셋은 [여기](https://www.dropbox.com/sh/i39abvnefllx2si/AAAbAYRvxzRp3cIE5HzqUw3ra?dl=0)에서 다운로드할 수 있습니다.

`data_lmdb_release.zip`에는 다음 데이터셋이 포함되어 있습니다:

- **훈련 데이터셋**:
  - [MJSynth (MJ)](http://www.robots.ox.ac.uk/~vgg/data/text/)
  - [SynthText (ST)](http://www.robots.ox.ac.uk/~vgg/data/scenetext/)
- **검증 데이터셋**:
  - [IC13](http://rrc.cvc.uab.es/?ch=2)
  - [IC15](http://rrc.cvc.uab.es/?ch=4)
  - [IIIT](http://cvit.iiit.ac.in/projects/SceneTextUnderstanding/IIIT5K.html)
  - [SVT](http://www.iapr-tc11.org/mediawiki/index.php/The_Street_View_Text_Dataset)
- **평가 데이터셋**:
  - [IIIT](http://cvit.iiit.ac.in/projects/SceneTextUnderstanding/IIIT5K.html)
  - [SVT](http://www.iapr-tc11.org/mediawiki/index.php/The_Street_View_Text_Dataset)
  - [IC03](http://www.iapr-tc11.org/mediawiki/index.php/ICDAR_2003_Robust_Reading_Competitions)
  - [IC13](http://rrc.cvc.uab.es/?ch=2)
  - [IC15](http://rrc.cvc.uab.es/?ch=4)
  - [SVTP](http://openaccess.thecvf.com/content_iccv_2013/papers/Phan_Recognizing_Text_with_2013_ICCV_paper.pdf)
  - [CUTE](http://cs-chan.com/downloads_CUTE80_dataset.html)

---

### 사전 학습된 모델로 데모 실행
1. [사전 학습된 모델](https://drive.google.com/drive/folders/15WPsuPJDCzhp2SvYZLRj8mAlT3zmoAMW)을 다운로드합니다.
2. 테스트할 이미지를 `demo_image/` 폴더에 추가합니다.
3. 아래 명령어를 실행하여 데모를 수행합니다.  
   **대소문자 구분 모델**을 사용하는 경우 `--sensitive` 옵션을 추가하세요.

```bash
CUDA_VISIBLE_DEVICES=0 python3 demo.py \
--Transformation TPS --FeatureExtraction ResNet --SequenceModeling BiLSTM --Prediction Attn \
--image_folder demo_image/ \
--saved_model TPS-ResNet-BiLSTM-Attn.pth

```

#### 예측 결과

| 데모 이미지 | [TRBA (**T**PS-**R**esNet-**B**iLSTM-**A**ttn)](https://drive.google.com/open?id=1b59rXuGGmKne1AuHnkgDzoYgKeETNMv9) | [TRBA (대소문자 구분)](https://drive.google.com/open?id=1ajONZOgiG9pEYsQ-eBmgkVbMDuHgPCaY) |
| ---         |     ---      |          --- |
| <img src="./demo_image/demo_1.png" width="300">    | available   | Available   |
| <img src="./demo_image/demo_2.jpg" width="300">      | shakeshack    | SHARESHACK    |
| <img src="./demo_image/demo_3.png" width="300">  | london   | Londen   |
| <img src="./demo_image/demo_4.png" width="300">      | greenstead    | Greenstead    |
| <img src="./demo_image/demo_5.png" width="300" height="100">    | toast   | TOAST   |
| <img src="./demo_image/demo_6.png" width="300" height="100">      | merry    | MERRY    |
| <img src="./demo_image/demo_7.png" width="300">    | underground   | underground  |
| <img src="./demo_image/demo_8.jpg" width="300">      | ronaldo    | RONALDO   |
| <img src="./demo_image/demo_9.jpg" width="300" height="100">    | bally   | BALLY  |
| <img src="./demo_image/demo_10.jpg" width="300" height="100">      | university    | UNIVERSITY    |

---

### 훈련 및 평가

1. **CRNN 모델 훈련**
```bash
CUDA_VISIBLE_DEVICES=0 python3 train.py \
--train_data data_lmdb_release/training --valid_data data_lmdb_release/validation \
--select_data MJ-ST --batch_ratio 0.5-0.5 \
--Transformation None --FeatureExtraction VGG --SequenceModeling BiLSTM --Prediction CTC
```

2. **CRNN 모델 평가** IC15-2077 평가를 수행하려면 데이터 필터링 부분을 확인하세요.  [data filtering part](https://github.com/clovaai/deep-text-recognition-benchmark/blob/c27abe6b4c681e2ee0784ad966602c056a0dd3b5/dataset.py#L148). 

```
CUDA_VISIBLE_DEVICES=0 python3 test.py \
--eval_data data_lmdb_release/evaluation --benchmark_all_eval \
--Transformation None --FeatureExtraction VGG --SequenceModeling BiLSTM --Prediction CTC \
--saved_model saved_models/None-VGG-BiLSTM-CTC-Seed1111/best_accuracy.pth
```

3. **TRBA 모델 훈련 및 평가** (**T**PS-**R**esNet-**B**iLSTM-**A**ttn) 최고 정확도 모델(TRBA)을 훈련 및 평가하려면 아래 명령어를 사용하세요.
사전 학습된 모델은 여기에서 다운로드할 수 있습니다. ([여기](https://drive.google.com/drive/folders/15WPsuPJDCzhp2SvYZLRj8mAlT3zmoAMW))
```
CUDA_VISIBLE_DEVICES=0 python3 train.py \
--train_data data_lmdb_release/training --valid_data data_lmdb_release/validation \
--select_data MJ-ST --batch_ratio 0.5-0.5 \
--Transformation TPS --FeatureExtraction ResNet --SequenceModeling BiLSTM --Prediction Attn
```

```
CUDA_VISIBLE_DEVICES=0 python3 test.py \
--eval_data data_lmdb_release/evaluation --benchmark_all_eval \
--Transformation TPS --FeatureExtraction ResNet --SequenceModeling BiLSTM --Prediction Attn \
--saved_model saved_models/TPS-ResNet-BiLSTM-Attn-Seed1111/best_accuracy.pth
```

### Arguments
* `--train_data`: 훈련용 LMDB 데이터셋 폴더 경로.
* `--valid_data`: 검증용 LMDB 데이터셋 폴더 경로.
* `--eval_data`: 평가용 LMDB 데이터셋 폴더 경로 (test.py에서 사용).
* `--select_data`: 훈련 데이터 선택. 기본값은 MJ-ST로, MJ와 ST 데이터를 훈련에 사용.
* `--batch_ratio`: 배치 내 각 데이터셋의 비율 설정. 기본값은 0.5-0.5로, 배치의 50%는 MJ, 나머지 50%는 ST 데이터로 구성.
* `--data_filtering_off`: [데이터 필터링](https://github.com/clovaai/deep-text-recognition-benchmark/blob/f2c54ae2a4cc787a0f5859e9fdd0e399812c76a3/dataset.py#L126-L146)을 생략하여 LmdbDataset 생성.
* `--Transformation`: Transformation 모듈 선택 [None | TPS].
* `--FeatureExtraction`: FeatureExtraction 모듈 선택 [VGG | RCNN | ResNet].
* `--SequenceModeling`: SequenceModeling 모듈 선택 [None | BiLSTM].
* `--Prediction`: Prediction 모듈 선택 [CTC | Attn].
* `--saved_model`: 평가에 사용할 저장된 모델 경로.
* `--benchmark_all_eval`: 논문 Table 1과 동일하게 10개의 평가 데이터셋 버전을 사용하여 평가.

---

## 실패 사례 및 정제된 라벨 다운로드
실패 사례 이미지와 정제된 라벨이 포함된 파일은 [여기](https://www.dropbox.com/s/5knh1gb1z593fxj/image_release_190624.zip?dl=0)에서 다운로드할 수 있습니다.

이 패키지에는 실패 사례 이미지와 정제된 라벨이 포함된 벤치마크 평가 이미지들이 포함되어 있습니다.

![Failure Cases](./figures/failure-case.jpg)


## 사용자 데이터셋 또는 비라틴어 데이터셋으로 훈련하려면

1. **LMDB 데이터셋 생성**  
   아래 명령어를 사용하여 사용자 데이터셋을 LMDB 형식으로 변환합니다.

```bash
pip3 install fire
python3 create_lmdb_dataset.py --inputPath data/ --gtFile data/gt.txt --outputPath result/

```
데이터 폴더의 구조는 다음과 같아야 합니다:
```
data
├── gt.txt
└── test
    ├── word_1.png
    ├── word_2.png
    ├── word_3.png
    └── ...
```
gt.txt 파일은 아래와 같은 형식을 따라야 합니다:
`{imagepath}\t{label}\n` <br>
gt.txt 파일의 예시는 다음과 같습니다:
```
test/word_1.png Tiredness
test/word_2.png kills
test/word_3.png A
...
```
2. **`--select_data`, `--batch_ratio`, 및 `opt.character` 수정**  
   사용자 데이터셋을 훈련하려면 이 옵션들을 적절히 수정해야 합니다.  
   자세한 내용은 [이 이슈](https://github.com/clovaai/deep-text-recognition-benchmark/issues/85)를 참조하세요.

---

## Acknowledgements
이 구현은 다음 레포지토리를 기반으로 하고 있습니다:
- [crnn.pytorch](https://github.com/meijieru/crnn.pytorch)
- [ocr_attention](https://github.com/marvis/ocr_attention)

---

## 참고 문헌
1. M. Jaderberg, K. Simonyan, A. Vedaldi, and A. Zisserman. Synthetic data and artificial neural networks for natural scenetext recognition. In Workshop on Deep Learning, NIPS, 2014.  
2. A. Gupta, A. Vedaldi, and A. Zisserman. Synthetic data for text localisation in natural images. In CVPR, 2016.  
3. D. Karatzas, F. Shafait, S. Uchida, M. Iwamura, L. G. i Bigorda, S. R. Mestre, J. Mas, D. F. Mota, J. A. Almazan, and L. P. De Las Heras. ICDAR 2013 robust reading competition. In ICDAR, pages 1484–1493, 2013.  
4. D. Karatzas, L. Gomez-Bigorda, A. Nicolaou, S. Ghosh, A. Bagdanov, M. Iwamura, J. Matas, L. Neumann, V. R. Chandrasekhar, S. Lu, et al. ICDAR 2015 competition on robust reading. In ICDAR, pages 1156–1160, 2015.  
5. A. Mishra, K. Alahari, and C. Jawahar. Scene text recognition using higher order language priors. In BMVC, 2012.  
6. K. Wang, B. Babenko, and S. Belongie. End-to-end scene text recognition. In ICCV, pages 1457–1464, 2011.  
7. S. M. Lucas, A. Panaretos, L. Sosa, A. Tang, S. Wong, and R. Young. ICDAR 2003 robust reading competitions. In ICDAR, pages 682–687, 2003.  
8. T. Q. Phan, P. Shivakumara, S. Tian, and C. L. Tan. Recognizing text with perspective distortion in natural scenes. In ICCV, pages 569–576, 2013.  
9. A. Risnumawan, P. Shivakumara, C. S. Chan, and C. L. Tan. A robust arbitrary text detection system for natural scene images. In ESWA, volume 41, pages 8027–8048, 2014.  
10. B. Shi, X. Bai, and C. Yao. An end-to-end trainable neural network for image-based sequence recognition and its application to scene text recognition. In TPAMI, volume 39, pages 2298–2304, 2017.  


## Links
- **Web Demo**: [https://demo.ocr.clova.ai/](https://demo.ocr.clova.ai/)  
  Clova AI의 검출 및 인식 기술을 결합한 데모로, 한국어 및 일본어에 대한 추가/고급 기능 포함.
- **Detection Repository**: [CRAFT-pytorch](https://github.com/clovaai/CRAFT-pytorch)

---