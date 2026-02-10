# TODO

## ~~1. 파일럿 노트북 실행 & 검증~~ (완료)
- [x] `1. Modeling.ipynb` XGB/CatBoost 섹션 실제 실행
- [x] adapter별 `evals_result` 키 이름 검증
- [x] 결과 비교 → 사용성 개선 필요 (→ 4번으로)

## 2. Stacking 워크플로우 완성
- [x] `_stacking.py`(레거시) 제거, `collector/_stacking.py`로 통일
- [x] `include_target=False`일 때 불필요한 target 빌드 제거
- [ ] 파일럿 노트북에서 StackingCollector로 LGB+XGB+CB 결과 수집
- [ ] 2nd level 모델: get_dataset → 새 Experimenter 생성

## 3. 테스트 코드 작성
- [ ] Pipeline 핵심 로직 unit test
- [ ] Experimenter 빌드/실험 흐름 test
- [ ] Collector 수집/조회 test

## 4. 프레임워크 사용성 개선
- [x] Node error 상태 구현 (build/exp 중 에러 시 error 상태 전환, 나머지 노드 계속 진행)
- [x] Experimenter 상태 요약 (`desc_status`: Stage/Head 상태 통계 + 에러 상세)
- [ ] 에러 분석 도구 (오분류 케이스 분석)

## 5. 패키지화 준비
- [ ] 패키지 이름 결정
- [ ] `pyproject.toml` 구성
- [ ] `__init__.py` 정리, 의존성 명시

## 6. 새로운 Collector 타입
- [x] OutputCollector (output_train/output_valid 저장, ConfusionMatrix 등 사후 분석 커버)
- ~~ConfusionMatrixCollector~~ → OutputCollector에서 사후 계산으로 대체
- [ ] ObjVarsCollector (fold별 입력/출력 변수 구성 수집, finalize 후에도 변수 정보 보존 → Report 정확성 확보)
- [ ] CalibrationCollector
