# CLAUDE 동작
니가 구사한 코드는 왠만한 건 다 파악 가능해. 주석은 나중에 한꺼번에 만들꺼야, 만들지마
함수나 메소드 가이드도 나중에 할꺼야.

CLAUDE.md에서 불필요하게 토큰을 낭비 하지 않도록, 작업 내역의 개요를 확인해라

# modeler 모듈 요약

## 핵심 클래스
- **Experimenter** (`_experimenter.py`): 실험 관리 중심 클래스
  - 생성자: `(data, path, sp, sp_v, splitter_params, title, data_key, logger)`
  - 구조 관리: `set_grp`, `set_node`, `rename_grp`, `remove_grp`, `remove_node`
  - 실행: `build` (pipe 노드), `exp` (exp 노드), `finalize`
  - 측정/스태킹: `add_metric`, `add_stacking`
  - 결과 분석: `get_result`, `get_results`, `get_results_merge` (adapter의 result_objs 활용)
  - 저장/로드: `_save`, `load`
  - 설명: `get_node_info` (Markdown 반환), `desc_spec`, `desc_pipeline`, `desc_node`, `desc_node_vars`
  - 저장: `{path}/__exp.pkl` (data 제외, data_key로 검증)

- **NodeGroup** (`_node.py`): 노드 그룹 (pipe/exp 역할), 계층 구조 (parent_grp/child_grps)
  - 저장: `{grp_path}/__grp.pkl`

- **Node** (`_node.py`): 개별 노드 (processor 실행 단위)
  - 저장: `{grp_path}/__{name}.pkl`, 빌드 결과 `{node_path}/obj{idx}.pkl`

- **RootNode** (`_node.py`): 원본 데이터 노드

- **Metric** (`_metric.py`): 실험 결과 측정
  - 저장: `{path}/__metric/{name}.pkl`

- **Stacking** (`_stacking.py`): 실험 결과 스태킹 (메모리 효율적)
  - 저장: `{path}/__stacking/{name}/__config.pkl`, 노드별 `{node}.pkl`

## Logger (`_logger.py`)
- **BaseLogger** (ABC): `info`, `warning`, `start_progress`, `update_progress`, `end_progress`, `adhoc_progress`, `clear_progress`
- **DefaultLogger**: 콘솔 출력 구현
  - progress: 스택 기반 다중 depth (`Build 1/3 > node_a 2/5 > 150/500 valid-rmse: 0.12`)
  - adhoc_progress: depth 추가 없이 현재 스택에 이어 붙임 (Early Stopping 등 중간 중단 가능 과정용)
  - clear_progress: 에러 시 스택 전체 정리
  - level 설정: `['info', 'warning', 'progress']` 조합

## 결과 분석 체계
- **adapter의 result_objs**: `{name: (callable, mergeable_bool)}`
  - callable: `(processor, **params) -> 결과` 형태의 정적 메소드
  - mergeable_bool: fold 간 집계 가능 여부
- **Experimenter.get_result(node, idx, result, params)**: 특정 fold의 결과 iterator
- **Experimenter.get_results(node, result, params)**: 전체 fold 결과
- **Experimenter.get_results_merge(node, result, params, agg_inner, agg_outer)**: fold 간 집계
- adapter별 result_objs:
  - sklearn: `coef`, `explained_variance`, `components`, `scalings`, `feature_importances`, `tree` 등
  - xgboost: `feature_importances`, `evals_result`, `trees`
  - lightgbm: `feature_importances_pvc`, `evals_result`, `trees`
  - catboost: `feature_importances_pvc`, `feature_importances_interaction`, `evals_result`, `trees`
- **analyzer**: ML 인스턴스만으로 분석 불가한 것 (SHAP 등) 용도로 별도 구현 예정

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
