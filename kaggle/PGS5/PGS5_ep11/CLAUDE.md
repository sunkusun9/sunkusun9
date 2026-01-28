# CLAUDE 동작
니가 구사한 코드는 왠만한 건 다 파악 가능해. 주석은 나중에 한꺼번에 만들꺼야, 만들지마
함수나 메소드 가이드도 나중에 할꺼야.

CLAUDE.md에서 불필요하게 토큰을 낭비 하지 않도록, 작업 내역의 개요를 확인해라

# modeler 모듈 요약

## 핵심 클래스
- **Experimenter** (`_experimenter.py`): 실험 관리 중심 클래스
  - 생성자: `(data, path, sp, sp_v, splitter_params, title, data_key)`
  - 주요 메소드: `add_grp`, `set_node`, `build`, `exp`, `add_metric`, `add_stacking`, `save`, `load`
  - 저장: `{path}/__exp.pkl` (data 제외, data_key로 검증)

- **NodeGroup** (`_node.py`): 노드 그룹 (pipe/exp 역할)
  - 저장: `{grp_path}/__grp.pkl`
  - `@classmethod load(experimenter, name, parent_grp)`

- **Node** (`_node.py`): 개별 노드 (processor 실행 단위)
  - 저장: `{grp_path}/__{name}.pkl`, 빌드 결과 `{node_path}/obj{idx}.pkl`
  - `@classmethod load(experimenter, grp, name)`

- **RootNode** (`_node.py`): 원본 데이터 노드

- **Metric** (`_metric.py`): 실험 결과 측정
  - 저장: `{path}/__metric/{name}.pkl`
  - `@classmethod load_from_file(experimenter, name)`

- **Stacking** (`_stacking.py`): 실험 결과 스태킹 (메모리 효율적)
  - 저장: `{path}/__stacking/{name}/__config.pkl`, 노드별 `{node}.pkl`
  - `@classmethod load_from_file(experimenter, name)`

## 보조 모듈
- **_data_wrapper.py**: DataWrapper (pandas/numpy 래핑, wrap/unwrap)
- **_node_processor.py**: TransformProcessor, PredictProcessor, resolve_columns
- **_describer.py**: desc_spec, desc_pipeline, desc_node, desc_node_vars
- **col.py**: 컬럼 선택 유틸리티
- **adapter/**: ML 프레임워크 어댑터 (sklearn, xgboost, lightgbm, catboost, keras)

## analyzer 모듈
- **BaseAnalyzer** (`analyzer/_base.py`): 분석기 추상 클래스
  - `requires_data`: X, y 필요 여부 (False가 기본)
  - `_start(node)`: 분석 시작 전 초기화
  - `_analyze(node, idx)`: idx 단위 분석 (내부에서 obj, X, y 접근)
  - `_end(node)`: 분석 마무리
- **Experimenter.analyze(analyzers, nodes)**: built 노드 대상 분석 실행

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
