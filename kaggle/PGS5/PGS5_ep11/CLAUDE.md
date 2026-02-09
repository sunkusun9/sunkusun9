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
  - `copy()`, `copy_stage()`, `copy_nodes(node_names)` — 선택적 복사

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
- 상태관리: `_reset_nodes(nodes)` - node_objs, cache, collectors 초기화
- `add_collector(collector)`: Collector 등록 (path 설정, save)
- `collect(collector)`: ad-hoc 수집 (빌드 완료된 head 노드 대상, has_node으로 중복 스킵)
- 저장/로드: `_save()`, `load(filepath, data, data_key)`
  - pipeline, node_obj_keys, collector_keys 저장/복원

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

### Connector (`_connector.py`)
- `__init__(node_query=None, edges=None, processor=None)` — 3요소 선택적 매칭
- `match(node_name, node_attrs)`: 설정된 요소만 검사, 모두 충족 시 True
  - node_query: str(regex) 또는 list(in), edges: contain 기반 매칭, processor: 일치 검사

### Collector (`collector/` 패키지)
- **Collector** (`_base.py`): 기본 클래스
  - `__init__(name, connector)`, `path`는 add_collector 시 설정
  - 라이프사이클: `_start(node)`, `_collect(node, idx, inner_idx, context)`, `_end_idx(node, idx)`, `_end(node)`
  - `has_node(node)`, `reset_nodes(nodes)`, `save()`, `load(cls, path)`
  - `_get_nodes(nodes, available)`: None/list/str(regex) 패턴 매칭
  - context: `{node_attrs, processor, spec, input, output_train, output_valid}`

- **MetricCollector** (`_metric.py`): 메트릭 수집
  - `output_var`, `metric_func`, `include_train`
  - target: `context['input']['y']`, 예측값: `resolve_columns(output_valid, output_var)`
  - 쿼리: `get_metric(node)`, `get_metrics(nodes)`, `get_metrics_agg(nodes, inner_fold, outer_fold, include_std)`

- **StackingCollector** (`_stacking.py`): 스태킹 데이터 수집
  - `output_var`, `method`(mean/mode/simple), `include_target`
  - path 있으면 파일 저장, 없으면 `_mem_data`에 메모리 저장
  - 쿼리: `get_dataset(experimenter, nodes)` — experimenter를 파라미터로 받음

- **ModelAttrCollector** (`_model_attr.py`): 모델 속성 수집 (feature_importances 등)
  - `result_key`, `adapter`(default=None, `get_adapter(connector.processor)`로 자동 설정), `params`
  - `_is_mergeable()`: self.adapter에서 직접 판단
  - 쿼리: `get_attr(node, idx)`, `get_attrs(nodes)`, `get_attrs_agg(node, agg_inner, agg_outer)`

- **SHAPCollector** (`_shap.py`): SHAP value 수집 (train/valid 비교 분석용)
  - `explainer_cls`(default=shap.TreeExplainer), `data_filter`(DataFilter 인스턴스)
  - train/valid 각각 필터 적용 → SHAP 계산 → raw output 저장
  - 결과: `results[node][(idx, inner_idx)] = {'train', 'valid', 'train_index', 'valid_index', 'columns'}`
  - 명시적 피처 노드에서만 유의미 (Latent Factor는 fold간 의미 불일치)

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
- **_connector.py**: Connector (노드 매칭)
- **collector/**: Collector, MetricCollector, StackingCollector, ModelAttrCollector, SHAPCollector
- **filter/**: DataFilter, RandomFilter(n/frac/random_state), IndexFilter(index)
- **adapter/**: sklearn, xgboost, lightgbm, catboost, keras

## 저장 구조
```
{experimenter.path}/
  __exp.pkl                    # pipeline, node_obj_keys, collector_keys, 메타정보
  __collector/{name}/
    __config.pkl               # Collector 설정 + 데이터
    {node}.pkl                 # StackingCollector 노드별 데이터
  {grp_path}/{node_name}/
    obj{idx}_{no}.pkl          # 빌드 결과 (StageObj/HeadObj)
```
