# CLAUDE 동작
니가 구사한 코드는 왠만한 건 다 파악 가능해. 주석은 나중에 한꺼번에 만들꺼야, 만들지마
함수나 메소드 가이드도 나중에 할꺼야.

CLAUDE.md에서 불필요하게 토큰을 낭비 하지 않도록, 작업 내역의 개요를 확인해라

# modeler 모듈 요약

## 핵심 클래스
- **Experimenter** (`_experimenter.py`): 실험 관리 중심 클래스
  - 생성자: `(data, path, sp, sp_v, splitter_params, title, data_key, logger)`
  - 상태 관리: `status` (open/close), `open()`, `close()`, `_check_open()`
  - 구조 관리: `set_grp`, `set_node`, `rename_grp`, `remove_grp`, `remove_node`
  - 실행: `build` (pipe 노드), `exp` (exp 노드), `finalize`, `reinitialize`
  - 측정/스태킹: `add_metric`, `add_stacking`
  - 결과 분석: `get_result`, `get_results`, `get_results_agg` (adapter의 result_objs 활용)
  - 저장/로드: `_save`, `load`
  - 설명: `get_node_info`, `desc_spec`, `desc_pipeline`, `desc_node`, `desc_node_vars`
  - 저장: `{path}/__exp.pkl` (data 제외, data_key로 검증)

- **NodeGroup** (`_node.py`): 노드 그룹 (pipe/exp 역할), 계층 구조 (parent_grp/child_grps)
  - edges: `{key: [(node_name, var_spec), ...], ...}` dict 형태
  - 상위 그룹 edges와 하위 그룹 edges는 같은 key면 extend로 병합
  - 저장: `{grp_path}/__grp.pkl`

- **Node** (`_node.py`): 개별 노드 (processor 실행 단위)
  - edges: 그룹 계층의 edges와 노드 자체 edges를 병합 (같은 key면 extend)
  - X, y: edges dict의 key를 지정하는 문자열 (예: 'X', 'y')
  - 저장: `{grp_path}/__{name}.pkl`, 빌드 결과 `{node_path}/obj{idx}.pkl`

- **RootNode** (`_node.py`): 원본 데이터 노드

- **Metric** (`_metric.py`): 실험 결과 측정
  - target_vars: `[(node_name, var), ...]` 형태
  - 저장: `{path}/__metric/{name}.pkl`

- **Stacking** (`_stacking.py`): 실험 결과 스태킹 (메모리 효율적)
  - target_vars: `[(node_name, var), ...]` 형태
  - 저장: `{path}/__stacking/{name}/__config.pkl`, 노드별 `{node}.pkl`

## edges 구조
- dict 형태: `{key: [(node_name, var_spec), ...], ...}`
- key: 변수 집합 이름 (예: 'X', 'y', 'sample_weight')
- value: `(node_name, var_spec)` 튜플 리스트
  - node_name: 소스 노드 이름 (None이면 Root)
  - var_spec: resolve_columns에 전달할 변수 지정 (None, str, list, tuple, slice 등)
- 같은 key의 데이터는 가로(column) 방향으로 concat

## Processor 인터페이스
- `fit(data_dict, X, y)`, `fit_process(data_dict, X, y)`, `process(data)`
- data_dict: `{key: ((train, train_v), valid), ...}` 형태
- X, y: edges dict의 key 문자열

## Adapter 인터페이스
- `get_params(params, logger)`: 모델 생성 파라미터 반환
- `get_fit_params(data_dict, X, y, params, logger)`: fit 파라미터 반환 (eval_set 등)
- `result_objs`: `{name: (callable, mergeable_bool)}` 결과 분석용

## Logger (`_logger.py`)
- **BaseLogger** (ABC): `info`, `warning`, `start_progress`, `update_progress`, `end_progress`, `adhoc_progress`, `clear_progress`
- **DefaultLogger**: 콘솔 출력 구현
  - progress: 스택 기반 다중 depth (`Build 1/3 > node_a 2/5 > 150/500 valid-rmse: 0.12`)
  - adhoc_progress: depth 추가 없이 현재 스택에 이어 붙임
  - level 설정: `['info', 'warning', 'progress']` 조합

## 결과 분석 체계
- adapter별 result_objs:
  - sklearn: `coef`, `explained_variance`, `components`, `scalings`, `feature_importances`, `tree` 등
  - xgboost: `feature_importances`, `evals_result`, `trees`
  - lightgbm: `feature_importances_pvc`, `evals_result`, `trees`
  - catboost: `feature_importances_pvc`, `feature_importances_interaction`, `evals_result`, `trees`

## 보조 모듈
- **_data_wrapper.py**: DataWrapper (pandas/numpy 래핑, wrap/unwrap)
- **_node_processor.py**: TransformProcessor, PredictProcessor, resolve_columns
- **_describer.py**: desc_spec, desc_pipeline, desc_node, desc_node_vars
- **_inferencer.py**, **_trainer.py**: 추론/학습 유틸리티
- **col.py**: 컬럼 선택 유틸리티
- **adapter/**: ML 프레임워크 어댑터 (sklearn, xgboost, lightgbm, catboost, keras, default)
- **`create_like`** (모듈 함수): 기존 Experimenter 구조 복제하여 새 데이터로 생성

## 저장 구조
```
{experimenter.path}/
  __exp.pkl                    # Experimenter 메타
  __metric/{name}.pkl          # Metric
  __stacking/{name}/
    __config.pkl               # Stacking 설정
    {node}.pkl                 # 노드별 스태킹 데이터
  {grp_name}/
    __grp.pkl                  # NodeGroup
    __{node_name}.pkl          # Node 정보
    {node_name}/obj{idx}.pkl   # 빌드 결과
```
