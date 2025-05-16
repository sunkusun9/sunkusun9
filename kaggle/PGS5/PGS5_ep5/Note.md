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
  >
  > 1. Calories를 대상변수로 n_estimators를 기준으로 과적합이 발생하기 전까지 늘려, 성능을 측정: n_estimators: 700, train_score: -0.0608, test-score: -0.0623
  > 2. Calories_log를 대상변수로 n_estimator를 기준으로 Calories 만큼의 train score가 나올 때까지 n_estimators를 늘려 성능 측정: n_estimator: 120, train_score: -0.608, test_score: -0.623
  > 3. Calories_log를 대상변수로 n_estimators를 기준으로 과적합이 발생하기 전까지 늘려, 성능을 측정: n_estimators: 700, train_score: -0.0560, test-score: -0.0609

**Task** 입력 변수 분석

- Height, Weight, Heart_Rate 에 이상치로 생각 되는 것이 보입니다. 성능과의 영향도를 살펴봅니다.

- Age, Height, Weight, Duration, Heart_Rate 는 정수라고 할 수 있습니다.

- Height, Weight는 상관도가 0.96으로 강한 상관도 가 있고, 그 외 변수와 상관도가 떨아집니다.

- Duration, Heart_Rate, Body_Temp는 강한 상관도 가 있습니다.

- Height, Weight는 Sex와 강한 상관도를 지니고 있습니다.

- Age, Duration, Heart_rate, Body_Temp 는 약한 상관도가 있습니다.

- 동일한 입력값이 관측되며, 동일 관측치별 Calories_Log의 표준 편차를 구해보니, 0.01 이 나옵니다. 내재된 RMSE는 0.01 근처가 되지 않을까 생각됩니다.

**Task** Learning Task와 검증 루틴 설정

- Target: Calories_log
- Org 포함 여부가 Calroies를 예측는 데 도움이 될 지 살펴 봅니다.
  > 원본데이터를 넣는 것은 도움은 되지 않고, 약간의 성능 저하를 야기하는 걸로 판단되어 제외를 결정합니다.
- 4-Fold Cross Validation으로 검증을 헀을 때의 결과가 Leader Board 상의 성능과의 차이를 살펴봅니다.
  > LinearRegression의 성능이 너무 떨어져, LGBMRegressor로 만들어 테스트 해봅니다.
  >
  > 제출시에는 원래의 Target의 역변환을 수행하고 최소값을 1이 되도록 클리핑을 합니다.
  >
  > CV4: 0.0605, Pubic Score: 0.5799 가 나옵니다.

**Task** 상관도 분석

- Sex와 Calories_log와 kruskal 분석: pvalue 2.2695e-29
- Body_Temp 0.95, Duration 0.94, Heart_Rate 0.88 강한 상관도
  > Body_Temp 만으로 예측 4F-CB: 0.292
  > 분석: 상관도는 높은
- Age: 0.118 약한 상관도
- Height: -0.042, Weight: -0.025 약한 상관도
- 트리기반으 모델이 월등히 성능이 좋은 걸로 보면, 속성간의 상호작이 강한 걸로 보입니다.

**Sub-Task**: CatBoost으로 속성간 Interaction을 조사합니다.

- Duration과 Heart_Rate와 강한 상호작용이 있는 것으로 나옵니다.
  > 두 변수와 강한 비례 관계에 의한 상호작용으로 보입니다.
- Duration / Heart_Rate / Body_Temp PCA로 차원 축소를 해보고 효과가 있을지 살펴봅니다.
  > PCA는 효과가 없습니다.
- 다른 사례에서 보았듯이 수치형 변수의 타겟인코Duration과 Heart_rate

**Sub-Task**: Duration과 Heart_Rate를 범주형 변수로 만들고, 두 결합된 변수를 Target Encoder를 사용하여 처리한 효과를 조사합니다.

**Sub-Task**: PolyNomial Transform의 효과성을 찾아봅니다.

- 연속형 변수간의 곱을 추가하여 LinearRegression에 적용했을 때, 효과성을 살펴봅니다.
- 2차 다항을 추가했을 때 CV가 0.097의 효과가 있었습니다.(O) CV: 0.0934
- 역수로 2차 다항을 구성하니 성능개선 효과가 더 보입니다. (O) CV: 0.0801
- 역수를 취한 변수를 추가할 때 성능 개선을 살펴봅니다. (X) CV: 0.1371
- 2차항에 하나는 역수를 곱한 3차항으로 성능 개선을 살펴봅니다. (O) CV: 0.0873
- 1차항에 2차항을 역수로 곱한 3차항으로 성능 개선을 살펴봅니다. (O) CV: 0.080
  > 다중 공선성이 있는 속성이 보입니다.
- 3차항으로 성능 개선을 살펴봅니다. (?) CV: 0.0998

> - 분석결과
>   다항식 변환은 효과성이 보입니다. 다항 변수를 만들고 속성 선택을 하여 변수를 추려냈을 때의 성능 개선 효과를 살펴볼만 합니다.

**Task** 사례 분석

| Article                                                                                                                              | Ver. | Feature                                                                                                                                                                                                                                | 검증법 | OOF    | Public  |
| ------------------------------------------------------------------------------------------------------------------------------------ | ---- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------ | ------ | ------- |
| [Only XGBoost](https://www.kaggle.com/code/jiaoyouzhang/calorie-only-xgboost)                                                        | V.4  | XGBoost 만을 사용, max_depth: 10, colsample_bytree=0.7, subsample=0.9, early_stopping을 건 strong learner<br/>깔끔함, XGB의 하이퍼파라미터 구성 참조                                                                                   | 5-Fold | 0.0599 | 0.05692 |
| [S05E05 \| Calorie Expenditure Prediction \| Ridge](https://www.kaggle.com/code/ravaghi/s05e05-calorie-expenditure-prediction-ridge) | V.16 | XGB, CB, LGB, LGB-goss(?), Autoglu-on, ... 5 fold oof predict에 Ridge를 Meta Model로 사용하여 Stacking<br/> 검증의 성능은 좋지만, 첫번째 Article 보다는 결과가 안나옴                                                                  | 5-Fold | 0.0590 | 0.05698 |
| [NN - MLP Starter - \[CV 0.0608\]](https://www.kaggle.com/code/cdeotte/nn-mlp-starter-cv-0-0608)                                     | V.1  | [32 swish Batch Norm, 64 swish Batch Norm, 32 swish Batch Norm, 1 ㅣlinear <br/> ReduceLOnPlateau, factor = 0.5, patience = 3 min_lr = 1e-6, EarlyStooping patienct = 10, restore_best_weights = True, mode = 'min'] 구조의 Perceptron | 5-Fold | 0.608  | 0.5824  |

- Sex를 Boolean 형태로 변환

- Public Score에 약간의 Bias가 있다고 생각이 듭니다.

- NN도 잘 동작합니다.

**Experiment**

1. 다항 속성 생성 및 선택

**그룹화된 단계적전진선택**

- 1차, 1 / 1차, 2차, 1차 / 1차, 2차 / 1차, 1차 / 2차, 3차 속성을 만듭니다.
- 전체 속성으로 선형 모델을 학습 시키고, 회귀계수를 봅니다.
- 단계가 끝나고 속성의 제외는 더 이상 제외해도 성능향상이 없을 때까지, 그리고 제외 됐던 변수는 제외에 포함되지 않게 구현됐습니다.

RMSE: 0.0681

이 과정에서 회귀 계수가 상당이 큰 변수들이 나왔습니다. 이를 제거할 필요가 보입니다.

- 변형된 전진 선택으로 성능 개선 효과를 보기는 쉽기 않아 보이고, 비효율 적이였습니다. 전체 변수가 300개는 변수의 수가 많았고, 선별할 여지가 많이 있다는
  점에서 효과적이지 않은 접근이었습니다. 일단은 회귀계수의 크기가 작은 것을 골라내어 봅니다.

- 회귀계수가 작은 것을 골라내는 것은 오히려 역효과가 있습니다.

- 공선성에 의해 회귀계수가 커진 변수들이 있습니다.

- 후진제거를 해봅니다.

- R2를 측정하여 공선성 
> GPU를 사용하여 구현을 했습니다. 하나의 변수를 제외하는 것에서 성능 개선효과를 볼 수 있었습니다.
>
- 다항 변환을 통해 나온 속성들이 강한 공선성을 지닌 게 많이 있습니다. 원 변수들의 공선성을 조사해봅니다. 
> 원래 변수들간에도 공선성이 큽니다. Height와 Duration을 제외하고 다항 파생변수를 만들었으며, 이를 사용하여 성능을 측정 했을때,
>
> Height를 제외 했을 때 0.07008, Height와 Duration을 제외했을 때는 0.07053이 나옵니다.
>
**발견**: OLS 기반으로 하는 선형회귀모델의 다중 공선성 조사는 굳이 선형회귀모델을 일일히 학습시킬 필요가 없습니다. Schur Complement을 이용하면 획기적으로 속도를 줄일 수 있습니다.
- 선형 모델 기반으로 전진 선택과 후진 제거 또한 비슷한 방법으로 진행 할 수 있습니다.

- Duration_log은 tarett과 강한 상관성을 보입니다. Duration 대신에 Duration_log를 다른 변수들과 조합하면 어떤지를 살펴 봅니다.

1. XGB, LGB, CB, NN, .. 등등의 ML 모델 생성 및 잔차 분석
   > Boost 모델에서는 Outlier와 공선성이 있음을 고려하여 컬럼 샘플링과 데이터포인트 샘플링의 효과성을 살펴볼 필요가 있습니다.
   > 상호작용
