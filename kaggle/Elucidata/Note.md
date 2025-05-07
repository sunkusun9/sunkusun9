# Metric

각데이터포인트의  C1 ~ C35와 예측값 간의 spearman correlation의 평균입니다.

# 탐색

## Target

- C1 ~ C35 타겟 변수는 모두 0이상이며 0으로 치우침이 강합니다.
> Log 변환을 하게 되면 치우침은 해소가 됩니다.


# 검증 설정

## 1. 6-Fold GroupKFold

목표는 주어지지 않은 slide에 대한 예측 결과를 뽑는 것입니다.

따라서, slide를 기준으로 GroupKFold를 사용했습니다. 모두 6개의 slide가 주어졌고 


# 데이터 처리

## Target 변수들을 Log 변환

C1 ~ C35를 로그변환하여 C1_l~C35_l 변수를 만듭니다.

## Tensorflow image pipeline

Elucid에서는 Train으로 2000x2000 에 가까운 6개의 slide의 Histology 이미지를 제공하고 있고 Train으로 약 8600개 spot으로 이루어져 있습니다.

Test로는 1개의 Histology 이미지를 제공하고 있으며 약 2000개의 spot을 예측점으로 하고 있습니다.

- 모두 2000x2000 이 되도록 padding을 해줍니다. 
- image 크기에 따른  spot을 중심으로 이미지 패치를 만듭니다.
- Rotation / Zoom / Contrast 등의 보강 과정을 넣어줍니다.


# 실험

## Regression C1 ~ C35

### Baseline

**실험 내용**

- 검증법: 6-Fold GroupKFold

- Group에 해당하는 slide를 제외하고, 나머지 slide의 C1~C35까지의 평균으로 예측합니다.

- 그리고 C1_l ~ C35_l 까지의 평균과 비교합니다. 

- Metric: 0.4380 으로 C1_l~C35_l이 C1~C35 보다 압도적으로 성능이 졸습니다.
  

### EfficientNetB0, B1 B2

**실험 내용**

- 검증법: 6-Fold GroupKFold 
  
- EfficientNetB0, B1, B2를 사용하여 MSE, MAE를 손실함수로 사용하여 35개의 변수를 예측하는 회귀 모델을 만들어 예측합니다.

모두 과적합이 발생했으면 **Baseline**에 비해 떨어지는 성능을 보입니다.

### EfficientNetB0 pretrain

**실험 내용**

[External Data 1](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE144240) External_images_1.ipynb
[External Data 2](https://www.github.com/almaan/her2st') External_image_2.ipynb 두 개의 데이터소스에서 Histology와 Spot 데이터를

가져왔고, Elucid에서 제공해주는 Histology 이미지 크기인 2000x2000으로 축소시켜 Elucid 약 1만 + 2만의 총 3만장의 224x224 이미지 패치를

만드는 이미지 pipeline 확보했습니다.

EfficientNetB0 를 Encoder로 사용하여, AutoEnccoder를 만들어 30,000 장의 이미지로 MSE를 손실함수로 하여 학습을 시킵니다.


### Regression based on EfficientNetB0 pretrain Model

**실험 내용**

- 사전학습 시킨 EfficientNetB0를 기반으로 회귀 모델을 만듭니다.

- 검증법: 6-Fold GroupKFold를 이용하여

### Classify slide with EfficientNetB0

**실험 내용**

- EfficientNetB0을 가지고 Test케이스로 포함됨 7개의 Slide의 이미지 패치들을 입력으로 어떤 slide의 패치인지 구분하는 모델을 만들어,
Slide 별로 지닌 데이터의 특징 유무를 파악합니다.

- Stratified Split with 80% train 20% test

**결과**

- Test에 대 한 결과는 99% 이상

**분석**

- 분류 정확도가 모두 99% 이상이라는 것은 이미지 별로 특이점이 강하며, 이미지별 특이점은 일반화 성능을 저해하는 큰 걸림돌로 작용하고 있습니다.

- 이미지 별 특이점이 아닌 일반화가 가능한 특성(Feature)를 뽑아내는 법이 필요합니다.


**인사이트**

- Target 변수들 모두 C1_l ~ C35_l 을 예측하는 것이 더욱 효과적 입니다.
