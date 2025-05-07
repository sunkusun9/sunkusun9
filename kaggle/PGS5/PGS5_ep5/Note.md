
# 주제: Predict calorie Expenditure

[Calories Burnt Prediction](https://www.kaggle.com/datasets/ruchikakumbhar/calories-burnt-prediction)를 모태로 나온 경연입니다.

원래 데이터 탐색을 해볼만합니다.

**Task** Org와 Train / Test 차이 분석

- 원본 데이터와 Competition 데이터를 구분하는 모델을 만들어 분포상 차이점을 확인해봅니다.

- Org / Test 를 분류하는 LGBMClassifier 모델을 만들어 성능을 측정
> 결과: AUC: Train 0.6932, Test: 0.6294
> 
> 분석: 약간의 변별력을 가진 다는 점에서 차이가 존재합니다.
>
> LGBMClassifier의 속성 중요도는 Sex를 제외하고는 비슷한 중요도를 보입니다.

- Org / Train를 분류하는 LGBMClassifier 모델을 만들어 성능을 측정
> 결과: AUC: Train: 0.6940, Test: 0.625
>
> 분석: Test 처럼 약간의 변별력을 지니며, 속성의 중요도 또한 비슷합니다. 


종합: Org 가 도움 여부는 이 실험에서는 판단할 수 없습니다. Calories를 예측 모델을 만드는 단계에서 Org를 추가 여부에 따른 성능차이를 측정하여,

포함여부를 살펴봅니다.

**Task**: Train / Test 차이 분석

- Train / Test를 구분하는 모델을 입력 데이터 만으로 만들어 차이점을 살펴 봅니다.

- Train / Test을 분류하는 LGBMClassifier 모델을 만들어 Stratified Shuffle Split (Train 비율 80%)을 사용하여 분류 성능을 측정

> 결과: AUC: Train: 0.5446, Test: 0.4994
> 
> 분석: 구분의 용이점은 벌견되지 않아, 서로 차이가 나지 않습니다.

**결론**: Train / Test 입력 변수의 구성 상에는 차이가 나지 않고, Train / Test의 비율이 4:1 이라는 점에서, 일단은 4-Fold Cross-Validation을 사용하여

예측을 합니다.

Metric: RMSLE

**Task**: Target 분석

RMSLE 이므로, Target을 Log 변환하는 것이 유리하리라 생각이 됩니다. 

- Log 변환 전후의 분포를 비교해봅니다.

분석 결과
> Calories는 좌측으로 치우친 분포를 하고 있습니다.
>
> Calories_log는 4개 ~ 4개의 봉우리가 있습니다. 봉우리에 관련된 요인을 찾는 것이 도움이 될 수 있으리라 생각됩니다. 
>
> 정수입니다.

- Log를 취하는 것이 성능을 높일 수 있을 지 살펴봅니다.
> LGBMRegressor 심플하게 만들고 Shuffle Split Train size 0.8로 잡아 1회 검증을 통해 결과의 차이를 비교합니다.
> Calories와 Calories_Log와 비교 결과
> 3개의 실험을 수행
> 1. Calories를 대상변수로 n_estimators를 기준으로 과적합이 발생하기 전까지 늘려, 성능을 측정: n_estimators: 700, train_score: -0.0608, test-score: -0.0623
> 2. Calories_log를 대상변수로 n_estimator를 기준으로 Calories 만큼의 train score가 나올 때까지 n_estimators를 늘려 성능 측정: n_estimator: 120, train_score: -0.608, test_score: -0.623
> 3. Calories_log를 대상변수로  n_estimators를 기준으로 과적합이 발생하기 전까지 늘려, 성능을 측정: n_estimators: 700, train_score: -0.0560, test-score: -0.0609

**Task** 입력 변수 분석

- Height, Weight, Heart_Rate 에 이상치로 생각 되는 것이 보입니다.  성능과의 영향도를 살펴봅니다.

- Age, 	Height,	Weight, Duration, Heart_Rate 는 정수라고 할 수 있습니다.

- Height, Weight는 상관도가 0.96으로 강한 상관도 가 있고, 그 외 변수와 상관도가 떨아집니다. 

- Duration, Heart_Rate, Body_Temp는 강한 상관도 가 있습니다.

- Height, Weight는 Sex와 강한 상관도를 지니고 있습니다.

- Age, Duration, Heart_rate, Body_Temp 는 약한 상관도가 있습니다.

- 동일한 입력값이 관측되며, 동일 관측치별 Calories_Log의 표준 편차를 구해보니, 0.01 이 나옵니다. 내재된 RMSE는 0.01 근처가 되지 않을까 생각됩니다.

**Task** Learning Task 설정을 위한 작업

- Org 포함 여부가 Calroies를 예측는 데 도움이 될 지 살펴 봅니다.

**Task** 상관도 분석

**Task** Learning Task 설정 <TODO>



- Target: Calories_log 


4-Fold Cross Validation으로 검증을 헀을 때의 결과가 Leader Board 상의 성능과의 차이를 살펴봅니다.

**Task** 사례 분석

**Experiment** 