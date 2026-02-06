# CLAUDE 동작
니가 구사한 코드는 왠만한 건 다 파악 가능해. 주석은 나중에 한꺼번에 만들꺼야, 만들지마
함수나 메소드 가이드도 나중에 할꺼야.

CLAUDE.md에서 불필요하게 토큰을 낭비 하지 않도록, 작업 내역의 개요를 확인해라

# modeler 모듈 요약

## 아키텍처 개요
- **Pipeline** (`_pipeline.py`): 노드 그래프 자료구조 (ML 관심사 분리)
- **Experimenter** (`_experimenter.py`): 실험 실행/관리 (Pipeline 사용)
- **ExpObj** (`_expobj.py`): 노드별 빌드/실험 객체 관리

## 핵심 클래스

### Pipeline 계층 (`_pipeline.py`)
- **Pipeline**: 노드 그래프 관리
  - `nodes`: `{name: PipelineNode}`, `grps`: `{name: PipelineGroup}`
  - `set_grp`, `set_node`, `rename_grp`, `remove_grp`, `remove_node`
  - `get_node_names(query)`, `get_node_attrs(name)`, `_get_effected_nodes(nodes)`

- **PipelineGroup**: 노드 그룹 (stage/head 역할)
  - 속성: `name`, `role`, `processor`, `edges`, `method`, `parent`, `adapter`, `params`
  - `children`: 자식 그룹명 리스트, `nodes`: 소속 노드명 리스트
  - `get_attrs(grps)`: 상위 그룹 속성 병합하여 반환

- **PipelineNode**: 개별 노드
  - 속성: `name`, `grp`, `processor`, `edges`, `method`, `adapter`, `params`
  - `output_edges`: 이 노드를 입력으로 사용하는 노드명 리스트
  - `get_attrs(grps)`: 그룹 속성과 노드 속성 병합

### Experimenter (`_experimenter.py`)
- 생성자: `(data, path, ..., cache_maxsize=4GB, logger)`
- `pipeline`: Pipeline 인스턴스
- `node_objs`: `{node_name: StageObj|HeadObj}`
- `cache`: DataCache (LRU, 용량 기반)
- 실행: `build(nodes)` (stage), `exp(nodes)` (head)
- 상태관리: `_reset_nodes(nodes)` - node_objs, cache, metric, stacking 초기화
- 저장/로드: `_save()`, `load(filepath, data, data_key)`
  - pipeline 객체 직접 저장, node_obj_keys로 복원

### DataCache (`_experimenter.py`)
- `cachetools.LRUCache` 기반, 용량(bytes) 단위 관리
- `get_data(node, typ, idx)`, `put_data(node, typ, idx, data)`
- `clear_nodes(nodes)`: 특정 노드들의 캐시 삭제

### ExpObj (`_expobj.py`)
- **StageObj**: stage 역할 노드의 빌드 객체
  - `load()`: 파일에서 objs_ 복원, 없으면 status='finalized'
  - `start_build()`, `build_idx()`, `end_build()`, `get_objs(idx)`, `finalize()`

- **HeadObj**: head 역할 노드의 실험 객체
  - `load()`: 파일 존재 여부로 status 복원
  - `start_exp()`, `exp_idx()`, `end_exp()`, `get_objs(idx)`, `finalize()`

### 측정/스태킹
- **Metric** (`_metric.py`): `target_vars`, `output_var`, `metric_func`
- **Stacking** (`_stacking.py`): `target_vars`, `output_var`, `method`

## edges 구조
- dict 형태: `{key: [(node_name, var_spec), ...], ...}`
- key: 변수 집합 이름 (예: 'X', 'y', 'sample_weight')
- 같은 key의 데이터는 column 방향으로 concat
- 상위→하위 병합: 같은 key면 extend

## Processor (`_node_processor.py`)
- **TransformProcessor**: `fit`, `fit_process`, `process`
- **PredictProcessor**: `fit`, `fit_process`, `process`
- `data_dict`: `{key: ((train, train_v), valid), ...}` 형태

## Adapter 인터페이스
- `get_params(params, logger)`: 모델 생성 파라미터
- `get_fit_params(data_dict, X, y, params, logger)`: fit 파라미터
- `result_objs`: `{name: (callable, mergeable_bool)}`

## 보조 모듈
- **_data_wrapper.py**: DataWrapper (wrap/unwrap)
- **_describer.py**: desc_spec, desc_pipeline, desc_node, desc_obj_vars
- **_logger.py**: BaseLogger, DefaultLogger
- **col.py**: 컬럼 선택 유틸리티
- **adapter/**: sklearn, xgboost, lightgbm, catboost, keras

## 저장 구조
```
{experimenter.path}/
  __exp.pkl                    # pipeline, node_obj_keys, 메타정보
  __metric/{name}.pkl          # Metric
  __stacking/{name}/
    __config.pkl               # Stacking 설정
    {node}.pkl                 # 노드별 스태킹 데이터
  {grp_path}/{node_name}/
    obj{idx}_{no}.pkl          # 빌드 결과 (StageObj/HeadObj)
```
