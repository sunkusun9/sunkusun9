# ML Pipeline Framework

Machine Learning 솔루션 개발을 위한 범용 파이프라인 프레임워크. 데이터 전처리부터 모델 학습, 추론까지의 전 과정을 DAG(Directed Acyclic Graph) 기반 파이프라인으로 관리한다.

**활용 범위:**
- 데이터 분석 대회 (Kaggle 등)
- 실무 ML 프로젝트
- 프로덕션 모델 배포
- 실험 재현성 관리

## 아키텍처 개요

```
┌─────────────────────────────────────────────────────────────┐
│                      Core Components                        │
├─────────────────────────────────────────────────────────────┤
│  Experimenter    │  Trainer        │  Inferencer            │
│  (실험/검증)      │  (학습)          │  (추론)                │
│  2-level split   │  1-level split   │  fit 없이 predict      │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                    Support Components                       │
├─────────────────────────────────────────────────────────────┤
│  Node/NodeGroup  │  Processor       │  DataWrapper          │
│  (파이프라인)     │  (처리 로직)      │  (데이터 추상화)       │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                    Utility Components                       │
├─────────────────────────────────────────────────────────────┤
│  adapter.py      │  col.py          │  data_wrapper.py      │
│  (ML 프레임워크)  │  (컬럼 선택)      │  (멀티 라이브러리)     │
└─────────────────────────────────────────────────────────────┘
```

---

## 핵심 코드

### 1. `experimenter.py` - 실험 관리자

유연한 데이터 분할 전략 기반의 실험 및 모델 검증을 관리하는 핵심 클래스.

**주요 기능:**
- **2-Level 데이터 분할**: Outer split(평가용) + Inner split(early stopping용)
- **DAG 기반 파이프라인**: Node 간 의존성을 그래프로 관리
- **그룹 관리**: NodeGroup으로 유사한 노드들을 묶어 일괄 설정

**지원 분할 전략** (sklearn splitter 인터페이스 호환):
| Splitter | 용도 |
|----------|------|
| `ShuffleSplit(n_splits=1)` | 단순 홀드아웃 (기본값) |
| `KFold`, `StratifiedKFold` | K-fold 교차 검증 |
| `TimeSeriesSplit` | 시계열 데이터 분할 |
| `GroupKFold`, `GroupShuffleSplit` | 그룹 기반 분할 |
| 커스텀 splitter | `split()` 메서드만 구현하면 사용 가능 |

**클래스:**
```python
class Experimenter:
    def __init__(self, data, data_names=None, sp=ShuffleSplit(n_splits=1), sp_v=None, splitter_params=None)
    # sp: outer splitter (평가용) - 기본값은 단순 홀드아웃
    # sp_v: inner splitter (early stopping용, 선택)
    # splitter_params: splitter에 전달할 파라미터 (예: {'y': 'target'})
```

**멤버 속성:**
| 속성 | 설명 |
|------|------|
| `sp` | outer splitter 객체 |
| `sp_v` | inner splitter 객체 |
| `splitter_params` | splitter 파라미터 dict |
| `train_idx_list` | 학습 인덱스 리스트 |
| `valid_idx_list` | 검증 인덱스 리스트 |
| `nodes` | 노드 딕셔너리 |
| `grps` | 그룹 딕셔너리 |

**핵심 메서드:**
| 메서드 | 설명 |
|--------|------|
| `add_grp()` | 노드 그룹 추가 |
| `set_grp()` | 그룹 설정 변경 (하위 노드 자동 재빌드) |
| `set_node()` | 파이프라인에 노드 추가/수정 |
| `remove_node()` | 노드 제거 |
| `split()` | 분할 전략별 데이터 제공 |
| `to_mermaid()` | 파이프라인 시각화 |
| `save()/load()` | 객체 저장/불러오기 |

**사용 예시:**
```python
from experimenter import Experimenter
from sklearn.model_selection import ShuffleSplit, StratifiedKFold, TimeSeriesSplit

# 1) 단순 홀드아웃 (기본)
exp = Experimenter(data=train_df)

# 2) Stratified K-fold (분류 문제)
exp = Experimenter(
    data=train_df,
    sp=StratifiedKFold(n_splits=5),
    splitter_params={'y': 'target'}
)

# 3) Nested split (outer 평가 + inner early stopping)
exp = Experimenter(
    data=train_df,
    sp=StratifiedKFold(n_splits=5),      # outer: 5-fold
    sp_v=ShuffleSplit(n_splits=1),       # inner: 홀드아웃
    splitter_params={'y': 'target'}
)

# 4) 시계열 분할
exp = Experimenter(
    data=train_df,
    sp=TimeSeriesSplit(n_splits=5)
)

# 노드 추가
exp.set_node('scaler', processor=StandardScaler,
             edges=[(None, None)], method='fit_transform')
```

---

### 2. `trainer.py` - 학습 관리자

프로덕션/배포용 모델을 학습하는 클래스. Experimenter의 검증된 파이프라인 구조를 그대로 복제하여 전체 데이터로 최종 모델 학습.

**주요 기능:**
- **단일 레벨 분할**: Inner split만 사용 (early stopping/hyperparameter tuning용)
- **Experimenter 구조 복제**: `create_from()` 함수로 동일한 파이프라인 사용

**클래스:**
```python
class Trainer:
    def __init__(self, data, data_names=None, sp=ShuffleSplit(n_splits=1), splitter_params=None)
    # sp: splitter 객체
    # splitter_params: splitter에 전달할 파라미터
```

**멤버 속성:**
| 속성 | 설명 |
|------|------|
| `sp` | splitter 객체 |
| `splitter_params` | splitter 파라미터 dict |
| `train_idx_list` | (train_idx, valid_idx) 튜플 리스트 |

**핵심 함수:**
```python
def create_from(exp, data, data_names=None, sp=None, select=None, splitter_params=None):
    """Experimenter의 구조를 복제하여 Trainer 생성

    Args:
        exp: 원본 Experimenter
        data: 학습 데이터
        select: 복제할 노드 리스트 (None이면 전체)
        splitter_params: splitter 파라미터 (None이면 원본 exp와 동일)
    """
```

**사용 예시:**
```python
from trainer import Trainer, create_from

# Experimenter 구조 복제 (splitter_params 자동 상속)
trainer = create_from(exp, train_df)

# splitter_params 직접 지정
trainer = create_from(exp, train_df, splitter_params={'y': 'target'})

# 특정 노드만 선택
trainer = create_from(exp, train_df, select=['final_model'])
```

---

### 3. `inferencer.py` - 추론 관리자

학습된 Trainer의 모델로 테스트 데이터 예측을 수행하는 클래스.

**주요 기능:**
- **fit 없는 추론**: 이미 학습된 모델 객체 사용
- **파이프라인 자동 적용**: 전처리부터 예측까지 일괄 처리

**클래스:**
```python
class Inferencer:
    def __init__(self, trainer, data, y_columns=None, fold_idx=0)
    # trainer: 학습된 Trainer
    # y_columns: 추론 데이터에서 제외할 컬럼
    # fold_idx: 사용할 fold 인덱스
```

**핵심 메서드:**
| 메서드 | 설명 |
|--------|------|
| `predict(node_name)` | 특정 노드의 예측 결과 반환 |
| `predict_all()` | 모든 노드의 예측 결과 dict 반환 |

**사용 예시:**
```python
from inferencer import Inferencer

# 추론기 생성
inf = Inferencer(trainer, test_df, y_columns='target')

# 예측
predictions = inf.predict('final_model')
```

---

## 유틸리티 코드

### 4. `adapter.py` - ML 프레임워크 어댑터

각 머신러닝 프레임워크별 `fit()` 파라미터(특히 `eval_set`) 처리를 통일된 인터페이스로 추상화.

**지원 프레임워크:**
| 어댑터 | 지원 모델 |
|--------|----------|
| `XGBoostAdapter` | XGBClassifier, XGBRegressor |
| `LightGBMAdapter` | LGBMClassifier, LGBMRegressor, LGBMRanker |
| `CatBoostAdapter` | CatBoostClassifier, CatBoostRegressor |
| `KerasAdapter` | KerasClassifier, KerasRegressor |
| `DefaultAdapter` | sklearn 기본 모델 |

**주요 설정:**
```python
class ModelAdapter:
    def __init__(self, eval_mode='both', verbose=0.1):
        # eval_mode: 'none', 'valid', 'both'
        # verbose: 0 (silent), 0-1 (%), >=1 (iteration)
```

**사용 예시:**
```python
from adapter import get_adapter, register_adapter

# 자동 어댑터 선택
adapter = get_adapter(XGBClassifier)

# 커스텀 어댑터 등록
register_adapter('MyModel', MyCustomAdapter())
```

---

### 5. `col.py` - 컬럼 선택 헬퍼

OneHotEncoder 등의 출력 컬럼을 처리하는 헬퍼 함수.

**함수:**
```python
def ohe_drop_first(columns, org_X):
    """OneHotEncoder 출력에서 각 변수의 첫 번째 더미 변수 제거

    다중공선성 방지를 위해 사용.
    """
```

**사용 예시:**
```python
from col import ohe_drop_first

# Node 설정 시 X 파라미터로 사용
exp.set_node('encoder',
             processor=OneHotEncoder,
             X=ohe_drop_first,  # 함수를 직접 전달
             ...)
```

---

### 6. `data_wrapper.py` - 데이터 추상화 래퍼

pandas, Polars, cuDF, NumPy 등 다양한 데이터 라이브러리를 통일된 인터페이스로 사용.

**지원 래퍼:**
| 클래스 | 대상 라이브러리 |
|--------|----------------|
| `PandasWrapper` | pandas DataFrame/Series |
| `PolarsWrapper` | Polars DataFrame |
| `CudfWrapper` | cuDF DataFrame (GPU) |
| `NumpyWrapper` | NumPy ndarray |

**공통 인터페이스:**
```python
class DataWrapper(ABC):
    def iloc(self, indices)        # 행 인덱싱
    def select_columns(self, columns)  # 컬럼 선택
    def get_columns(self)          # 컬럼명 리스트
    def get_shape(self)            # (rows, cols)
    def get_index(self)            # 인덱스
    def concat(wrappers, axis)     # 결합 (static)
    def from_output(output, columns, index)  # 출력 변환
```

**편의 함수:**
```python
from data_wrapper import wrap, unwrap

# 자동 래핑
wrapped = wrap(pandas_df)  # → PandasWrapper

# 네이티브 객체 추출
native = unwrap(wrapped)   # → pandas.DataFrame
```

---

## 지원 코드

### `node.py` - 파이프라인 노드

- `Node`: 단일 처리 단계 (전처리기, 모델 등)
- `NodeGroup`: 유사한 노드들의 그룹 (공통 설정 상속)
- `RootNode`: 원본 데이터를 제공하는 루트 노드

### `processor.py` - 처리 로직

- `TransformProcessor`: transform 계열 처리 (StandardScaler, OneHotEncoder 등)
- `PredictProcessor`: predict 계열 처리 (분류기, 회귀기 등)
- `resolve_columns()`: X, y 파라미터를 실제 컬럼 리스트로 변환

---

## 전체 워크플로우

```
1. 실험 (Experimenter)
   ├── 데이터 로드
   ├── 파이프라인 구성 (add_grp, set_node)
   ├── 분할 전략에 따른 성능 검증
   └── 최적 파이프라인 확정

2. 학습 (Trainer)
   ├── Experimenter 구조 복제 (create_from)
   ├── 전체 데이터로 최종 모델 학습
   └── 모델 저장 (save)

3. 추론 (Inferencer)
   ├── Trainer 로드
   ├── 신규 데이터로 Inferencer 생성
   └── predict()로 예측 결과 추출

4. 배포 (프로덕션)
   ├── Trainer/Inferencer pickle 파일 배포
   ├── API 서버에서 Inferencer.load() 후 predict()
   └── 배치 추론 또는 실시간 추론
```

---

## 파일 구조

```
PGS5_ep11/
├── experimenter.py    # 실험/검증 (2-level split)
├── trainer.py         # 학습 (1-level split)
├── inferencer.py      # 추론 (fit 없음)
├── node.py            # Node, NodeGroup, RootNode
├── processor.py       # TransformProcessor, PredictProcessor
├── adapter.py         # ML 프레임워크 어댑터
├── col.py             # 컬럼 선택 헬퍼
└── data_wrapper.py    # 멀티 라이브러리 래퍼
```

---

## 설계 철학

**1. 실험-학습-추론 분리**
- 실험 단계에서 검증된 파이프라인을 학습/추론에서 그대로 재사용
- 코드 중복 없이 일관된 전처리 보장

**2. 확장 가능한 구조**
- Adapter 패턴으로 새로운 ML 프레임워크 추가 용이
- DataWrapper로 다양한 데이터 라이브러리 지원 (GPU 가속 포함)
- NodeGroup 상속으로 복잡한 파이프라인 계층화

**3. 재현성 보장**
- 파이프라인 전체를 pickle로 저장/복원
- DAG 구조로 처리 순서 명확히 정의
- Mermaid 다이어그램으로 파이프라인 시각화
