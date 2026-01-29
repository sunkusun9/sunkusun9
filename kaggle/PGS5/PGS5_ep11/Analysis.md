# modeler 프레임워크 분석

## 1. 개요

modeler는 **DAG 기반 중첩 교차검증(Nested CV) ML 실험 프레임워크**이다. Kaggle 등 테이블 데이터 경진대회에서 반복되는 전처리-학습-평가-스태킹 워크플로우를 체계적으로 관리하기 위해 설계되었다.

## 2. 실험 라이프사이클

```
Data → Experimenter(data, path, sp, sp_v)
         │
         ├─ set_grp(): 노드 그룹 정의 (pipe / exp 역할)
         ├─ set_node(): DAG 구성 (edges로 노드 간 의존성)
         │
         ├─ build(): pipe 노드 fit → obj{idx}.pkl 캐싱
         ├─ exp(): exp 노드 학습 + metric/stacking 수집
         │
         ├─ get_results_merge(): adapter result_objs 기반 결과 분석
         └─ Stacking.get_dataset(): OOF 예측값 집계
```

## 3. 핵심 설계

### 3.1 이중 분할 구조 (sp + sp_v)

- **sp (바깥 fold)**: 모델 평가용. 최종 성능 측정에 사용
- **sp_v (안쪽 fold)**: early stopping, 하이퍼파라미터 튜닝용
- `train_idx_list[outer][inner] = (train_idx, valid_idx)` 형태의 중첩 인덱스

바깥 fold의 validation은 학습에 전혀 관여하지 않으므로, **과적합 없는 성능 추정**과 **OOF 예측값 생성**이 동시에 가능하다.

### 3.2 pipe / exp 역할 분리

| 역할 | 용도 | 실행 | 캐싱 |
|------|------|------|------|
| **pipe** | 전처리 파이프라인 | `build()` | fit 결과를 디스크에 저장, 재사용 |
| **exp** | 실험용 모델 | `exp()` | 매번 재학습, metric/stacking 수집 후 `finalize()`로 정리 |

pipe 노드는 한번 build하면 캐싱되어 exp 노드가 여러 번 재실행되어도 전처리를 반복하지 않는다.

### 3.3 DAG 기반 파이프라인

- `edges = [(node_name, var)]`로 노드 간 의존성 정의
- `output_edges`로 역방향 탐색 → BFS 기반 사이클 검출
- `_get_effected_nodes`로 변경 영향 범위 및 재빌드 우선순위 결정
- NodeGroup 계층 구조로 공통 설정(processor, edges, X, y, method, params) 상속

### 3.4 Adapter 패턴

`ModelAdapter` 인터페이스로 XGBoost, LightGBM, CatBoost, Keras, sklearn의 차이를 흡수한다.

- `get_fit_params()`: eval_set, validation_data 등 프레임워크별 검증 데이터 전달 방식 통일
- `get_params()`: callback 주입 (progress 등)
- `result_objs`: 프레임워크별 결과 추출 (feature_importances, evals_result, trees 등)

### 3.5 DataWrapper 추상화

pandas, numpy, polars, cudf를 통일된 인터페이스(`iloc`, `select_columns`, `concat`, `from_output`)로 래핑한다. processor 경계에서 `wrap()`/`unwrap()` 변환이 일어나며, 출력 타입이 자동 전파된다.

### 3.6 Generator 기반 Lazy Evaluation

모든 데이터 흐름(`get_data`, `get_data_train`, `get_data_valid`)이 generator이다. Node에 triple cache 시스템으로 반복 접근을 최적화한다.

### 3.7 메모리 효율적 Stacking

- 실험 중 numpy array만 보관
- `_end()`에서 즉시 디스크 저장 + 메모리 해제
- `get_dataset()`에서 on-demand 로드
- inner fold 집계 방식: mean, mode, simple 선택

### 3.8 Logger

스택 기반 다중 depth progress로 중첩 루프의 진행 상황을 한 줄로 표현한다.

```
Build 1/3 (33%) > node_a 2/5 (40%) > 150/500 (30%) valid-rmse: 0.1234
```

- `adhoc_progress`: depth 추가 없이 현재 스택에 이어 붙임 (Early Stopping 등 중간 중단 가능 과정용)
- `clear_progress`: 에러 시 스택 전체 정리 후 traceback 출력

## 4. 모듈 구성

### 핵심 클래스

| 클래스 | 파일 | 역할 |
|--------|------|------|
| Experimenter | `_experimenter.py` | 실험 전체 오케스트레이션 |
| NodeGroup | `_node.py` | 노드 그룹 (설정 상속, 계층 구조) |
| Node | `_node.py` | 개별 처리 단위 (processor 실행) |
| RootNode | `_node.py` | 원본 데이터 제공 |
| Metric | `_metric.py` | 실험 결과 측정 |
| Stacking | `_stacking.py` | OOF 예측값 수집 및 집계 |

### Processor

| 클래스 | 메소드 지원 | 출력 |
|--------|------------|------|
| TransformProcessor | transform, fit_transform | 변환된 feature |
| PredictProcessor | predict, predict_proba, fit_predict | 예측값/확률 |

### Adapter

| 어댑터 | 대상 | result_objs |
|--------|------|-------------|
| XGBoostAdapter | XGBClassifier/Regressor | feature_importances, evals_result, trees |
| LightGBMAdapter | LGBMClassifier/Regressor | feature_importances_pvc, evals_result, trees |
| CatBoostAdapter | CatBoost* | feature_importances_pvc/interaction, evals_result, trees |
| LMAdapter | LinearRegression 등 | coef |
| PCAAdapter | PCA | explained_variance, components |
| DecisionTreeAdapter | DecisionTree* | feature_importances, tree |
| KerasAdapter | Keras models | (미구현) |

### 보조 모듈

| 모듈 | 역할 |
|------|------|
| `_data_wrapper.py` | 데이터 라이브러리 추상화 |
| `_node_processor.py` | processor 래핑 (fit/transform/predict 통일) |
| `_describer.py` | Mermaid 다이어그램, Markdown 시각화 |
| `_logger.py` | 스택 기반 progress, warning 관리 |
| `_inferencer.py` | 학습된 모델 추론 전용 |
| `_trainer.py` | 단일 레벨 학습 (바깥 fold 없음) |
| `col.py` | 컬럼 선택 유틸리티 |
| `create_like` | Experimenter 구조 복제 |

## 5. 저장 구조

```
{experimenter.path}/
  __exp.pkl                          # 메타데이터 (data 제외, data_key 검증)
  __metric/{name}.pkl                # 노드별 metric 결과
  __stacking/{name}/
    __config.pkl                     # Stacking 설정
    {node}.pkl                       # 노드별 OOF 예측값
  {grp_name}/
    __grp.pkl                        # NodeGroup 설정
    __{node_name}.pkl                # Node 메타데이터
    {node_name}/obj{idx}.pkl         # 빌드된 processor (outer fold별)
```

## 6. 기존 도구와의 비교

### 6.1 sklearn Pipeline / FeatureUnion

sklearn의 Pipeline은 **선형 체인** 구조만 지원한다. FeatureUnion으로 병렬 분기를 만들 수 있지만, 임의의 DAG(특정 노드의 출력을 여러 하위 노드에서 참조)는 불가능하다. 또한 중간 단계는 반드시 transformer여야 하므로, predictor를 파이프라인 중간에 배치하는 것이 구조적으로 어렵다.

modeler는 `edges`로 임의의 DAG를 구성하며, transformer와 predictor를 구분 없이 파이프라인의 어느 위치에든 배치할 수 있다.

### 6.2 sklearn StackingClassifier / StackingRegressor

sklearn의 StackingClassifier는 `cross_val_predict`로 OOF 예측을 생성하지만, **base estimator는 전체 데이터에 대해 다시 fit**된다. 즉 OOF 생성과 최종 모델 학습이 별개의 과정이다.

modeler의 Stacking은 실험(exp) 과정에서 자연스럽게 OOF 예측을 수집하므로, 불필요한 재학습이 없다. inner fold 집계(mean/mode/simple)도 내장되어 있으며, 메모리 효율을 위해 디스크 기반으로 동작한다.

### 6.3 MLflow

MLflow는 **실험 추적(tracking)** 도구이다. 파라미터, 메트릭, 아티팩트를 로깅하고 UI로 비교하는 것이 핵심이다. 하지만 **실험의 실행 자체**(데이터 분할, 파이프라인 구성, 교차검증)는 사용자가 직접 구현해야 한다.

modeler는 실험의 실행 자체를 관리한다. 분할, 파이프라인, 교차검증, 결과 수집이 프레임워크 내에서 일어난다. MLflow가 "무엇을 기록할지"를 다룬다면, modeler는 "무엇을 실행할지"를 다룬다.

### 6.4 Optuna

Optuna는 **하이퍼파라미터 최적화** 프레임워크이다. trial 단위로 objective function을 반복 실행하며 최적 파라미터를 탐색한다.

modeler는 하이퍼파라미터 탐색이 아니라 **주어진 설정으로 실험을 체계적으로 실행**하는 데 초점이 맞춰져 있다. Optuna의 trial 내부에서 modeler를 사용하는 것이 자연스러운 조합이다.

### 6.5 비교 요약

| 관심사 | sklearn Pipeline | MLflow | Optuna | **modeler** |
|--------|-----------------|--------|--------|-------------|
| 파이프라인 구조 | 선형 체인 | - | - | **임의 DAG** |
| 중첩 교차검증 | 수동 구현 | - | - | **내장 (sp + sp_v)** |
| OOF Stacking | StackingClassifier (재학습 필요) | - | - | **실험 중 자동 수집** |
| 전처리 캐싱 | - | - | - | **pipe 노드 캐싱** |
| 다중 프레임워크 | sklearn only | 로깅만 | - | **Adapter 패턴** |
| eval_set 자동화 | - | - | - | **inner fold → eval_set** |
| 실험 추적 | - | 핵심 기능 | trial 단위 | Logger (경량) |
| 파라미터 탐색 | GridSearchCV | - | 핵심 기능 | - |

## 7. 효용성

### 7.1 Kaggle 경진대회 워크플로우 최적화

Kaggle 테이블 데이터 대회에서의 전형적인 워크플로우는:

1. 전처리 파이프라인 구성 (결측치, 인코딩, 스케일링)
2. 여러 모델로 교차검증 실험
3. OOF 예측으로 스태킹/앙상블
4. feature importance 등 결과 분석

이 각 단계가 modeler에서 `set_grp`/`set_node` → `build` → `exp` → `get_results_merge`/`get_dataset`으로 대응된다. 특히 **전처리를 한번 build하면 캐싱**되므로, 모델만 바꿔가며 실험할 때 전처리를 반복하지 않는다.

### 7.2 inner fold의 이중 활용

sp_v로 생성된 inner fold는 두 가지 목적으로 동시에 사용된다:

- **학습 중**: eval_set으로 전달되어 early stopping에 활용
- **평가 시**: OOF 예측 생성의 기반

이를 통해 early stopping과 OOF stacking이 하나의 실험 루프에서 자연스럽게 결합된다.

### 7.3 DAG의 유연성

edges 기반 DAG 구조로, 하나의 전처리 노드 출력을 여러 모델 노드가 공유하거나, 여러 전처리 결과를 하나의 모델에 합쳐서 입력하는 것이 선언적으로 가능하다. NodeGroup의 계층 구조로 공통 설정을 상속하면서 개별 노드에서 override할 수 있다.

### 7.4 재현성과 영속성

- `data_key`로 데이터 무결성 검증
- 구조(그룹, 노드, 메트릭, 스태킹)가 디스크에 저장되어 `load()`로 복원 가능
- `create_like`로 동일 구조를 새 데이터에 적용 가능

### 7.5 결과 분석 체계

adapter의 `result_objs`를 통해 프레임워크별 결과(feature importance, learning curve, tree 구조)를 통일된 인터페이스(`get_result`, `get_results_merge`)로 접근한다. fold 간 집계(inner/outer)도 내장되어 있어, 안정적인 feature importance 추정 등이 가능하다.

## 8. 발전 방향 및 개선사항

### 8.1 Analyzer 확장

현재 결과 분석은 adapter의 `result_objs`를 통해 **ML 인스턴스 내부 속성**에 접근하는 방식이다. 하지만 SHAP, Partial Dependence, Permutation Importance 등은 모델 인스턴스만으로는 계산할 수 없고 입력 데이터가 함께 필요하다. 이를 위한 별도의 analyzer 모듈을 구현하여, built 노드에 대해 데이터와 모델을 함께 활용하는 분석을 지원할 수 있다.

- SHAP value 계산 (TreeExplainer, KernelExplainer 등)
- Partial Dependence Plot 데이터 생성
- Permutation Importance (모델 외부에서 feature 교란 기반 측정)
- fold 간 SHAP 집계 → 안정적인 feature 기여도 추정

### 8.2 병렬 처리

현재 build와 exp의 반복문은 순차 실행이다. 노드 간 의존성이 없는 경우(같은 depth의 노드들) 병렬 실행이 가능하다.

- `_get_effected_nodes`의 우선순위가 같은 노드들은 독립적 → 병렬 빌드 가능
- inner fold 간 병렬화 (각 fold의 fit은 독립적)
- joblib, multiprocessing, 또는 Ray 기반 구현

### 8.3 하이퍼파라미터 탐색 통합

현재는 Optuna 등 외부 도구와 조합하여 사용하는 구조이지만, Experimenter 레벨에서 파라미터 탐색을 지원하면 더 효율적이다.

- `set_node`의 params에 탐색 공간 정의
- pipe 캐싱을 활용한 효율적 탐색 (전처리 반복 제거)
- trial 간 공유 가능한 빌드 결과 재사용

### 8.4 실험 추적 연동

경량 Logger로 충분한 경우가 많지만, 대규모 실험에서는 MLflow 등 외부 추적 시스템과 연동하면 유용하다.

- BaseLogger를 구현한 MLflowLogger: metric, params를 MLflow에 자동 로깅
- 실험 간 비교, 히스토리 관리
- 아티팩트(모델, 스태킹 데이터) 연동

### 8.5 에러 복구와 부분 재실행

현재 build/exp 중 에러가 발생하면 progress를 정리하고 traceback을 보여주지만, 이미 완료된 fold의 결과는 활용되지 않는다.

- fold 단위 체크포인팅: 완료된 fold는 저장하고, 실패 지점부터 재개
- exp 노드별 독립 실행: 특정 노드만 재실험 (다른 노드 결과 유지)

### 8.6 Stacking 고도화

현재 1단계 스태킹을 지원하며, 다단계 스태킹은 새로운 Experimenter를 만들어 수동으로 구성해야 한다.

- 다단계 스태킹 자동화: stacking 결과를 다음 레벨의 입력으로 자동 연결
- stacking 결과에 대한 추가 전처리 (정규화, feature selection) 파이프라인 지원

### 8.7 시각화 강화

`_describer.py`가 Mermaid 다이어그램과 Markdown 테이블을 생성하지만, 실험 결과의 시각화는 사용자 몫이다.

- metric 결과의 fold별 분포 시각화
- learning curve (evals_result) 자동 플롯
- feature importance 비교 차트
- 노드 간 성능 비교 대시보드

### 8.8 테스트와 문서화

- 단위 테스트: 핵심 로직(사이클 검출, 우선순위 계산, 데이터 흐름)에 대한 테스트
- 통합 테스트: 전체 라이프사이클(build → exp → metric → stacking) 검증
- API 문서: 주요 클래스/메소드의 사용 가이드
- 튜토리얼: 실제 Kaggle 데이터셋을 사용한 end-to-end 예제
