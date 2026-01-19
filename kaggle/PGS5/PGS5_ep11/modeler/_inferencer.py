import pickle as pkl
from _data_wrapper import wrap, unwrap
from _node_processor import resolve_columns

class InferenceTransformProcessor():
    """추론 전용 TransformProcessor - fit 없이 transform만 수행"""
    def __init__(self, node, fitted_obj, X_, output_vars):
        self.node = node
        self.obj = fitted_obj  # 이미 fit된 객체
        self.X_ = X_  # 이미 resolve된 컬럼 리스트
        self.output_vars = output_vars

    def process(self, data):
        # DataWrapper에서 native로 변환
        data_X = unwrap(data.select_columns(self.X_))
        data_index = data.get_index()

        result = self.obj.transform(data_X)

        # data의 Wrapper 타입으로 변환
        data_wrapper_class = type(data)
        return data_wrapper_class.from_output(result, self.output_vars, data_index)

class InferencePredictProcessor():
    """추론 전용 PredictProcessor - fit 없이 predict만 수행"""
    def __init__(self, node, fitted_obj, X_, output_vars, method='predict'):
        self.node = node
        self.obj = fitted_obj  # 이미 fit된 객체
        self.X_ = X_  # 이미 resolve된 컬럼 리스트
        self.output_vars = output_vars
        self.method = method

    def process(self, data):
        # DataWrapper에서 native로 변환
        data_X = unwrap(data.select_columns(self.X_))
        data_index = data.get_index()

        if self.method == 'predict':
            predictions = self.obj.predict(data_X)
            column_names = self.output_vars

        elif self.method == 'predict_proba':
            if not hasattr(self.obj, 'predict_proba'):
                raise Exception(f"Model does not support predict_proba")

            predictions = self.obj.predict_proba(data_X)
            column_names = self.output_vars

        else:
            raise ValueError(f"Unknown method: {self.method}. Use 'predict' or 'predict_proba'")

        # data의 Wrapper 타입으로 변환
        data_wrapper_class = type(data)
        return data_wrapper_class.from_output(predictions, column_names, data_index)

class InferenceNode():
    """추론 전용 Node - fit 없이 이미 학습된 객체들 사용"""
    def __init__(self, inferencer, name, trainer_node, fold_idx=0):
        self.inferencer = inferencer
        self.name = name
        self.trainer_node = trainer_node
        self.fold_idx = fold_idx  # Trainer의 어느 fold를 사용할지
        self.method = trainer_node.method
        self.edges = trainer_node.edges
        self.processors = []

        # Trainer의 해당 fold에서 fit된 객체들 가져오기
        if fold_idx >= len(trainer_node.objs_):
            raise ValueError(f"fold_idx {fold_idx} out of range for node '{name}'")

        fold_objs = trainer_node.objs_[fold_idx]

        for obj, _, info in fold_objs:
            # 이미 fit된 객체를 사용하는 InferenceProcessor 생성
            if self.method in ['transform', 'fit_transform']:
                inference_proc = InferenceTransformProcessor(
                    self,
                    obj.obj,  # fit된 transformer
                    obj.X_,
                    obj.output_vars
                )
            else:  # predict, predict_proba, fit_predict
                inference_proc = InferencePredictProcessor(
                    self,
                    obj.obj,  # fit된 estimator
                    obj.X_,
                    obj.output_vars,
                    method=self.method if self.method in ['predict', 'predict_proba'] else 'predict'
                )

            self.processors.append(inference_proc)

    def get_data(self, v=None):
        """전체 데이터를 처리하여 반환"""
        it = self.inferencer.get_data(self.edges)

        results = []
        for data, proc in zip(it, self.processors):
            result = proc.process(data)

            # 필요하면 컬럼 필터링
            if v is not None:
                X = resolve_columns(result, v, org_X=proc.X_)
                result = result.select_columns(X)

            results.append(result)

        # 여러 fold의 결과를 평균내거나 concat
        if len(results) == 1:
            return results[0]
        else:
            # 기본적으로 첫 번째 fold만 반환 (필요시 평균 로직 추가 가능)
            return results[0]

class InferenceRootNode():
    """추론 전용 RootNode - 전체 데이터 제공"""
    def __init__(self, inferencer, data):
        self.inferencer = inferencer
        self.data = data

    def get_data(self, v=None):
        """전체 데이터를 그대로 반환"""
        if v is None:
            yield self.data
        else:
            yield self.data.select_columns(v)

class Inferencer():
    """추론 전용 클래스 - Trainer의 학습된 모델로 추론만 수행

    fit 기능 없이 이미 학습된 Trainer의 구조와 모델을 사용하여 추론 수행.
    """
    def __init__(self, trainer, data, y_columns=None, fold_idx=0):
        """
        Args:
            trainer: 학습된 Trainer 인스턴스
            data: 추론할 데이터
            y_columns: 제외할 y 컬럼 리스트 (None이면 제외 안함)
            fold_idx: Trainer의 어느 fold를 사용할지 (기본: 0)
        """
        self.trainer = trainer
        self.fold_idx = fold_idx

        # 데이터 준비
        data = wrap(data)

        # y_columns가 주어진 경우 제외
        if y_columns is not None:
            if isinstance(y_columns, str):
                y_columns = [y_columns]

            available_columns = data.get_columns()
            filtered_columns = [col for col in available_columns if col not in y_columns]
            data = data.select_columns(filtered_columns)
            print(f"🗑️  Excluded {len(y_columns)} y column(s): {y_columns}")

        self.data = data
        self.nodes = {None: InferenceRootNode(self, data)}

        # Trainer의 노드들을 InferenceNode로 변환
        print(f"🔄 Creating inference nodes from Trainer (fold_idx={fold_idx})...")
        node_count = 0
        for name, trainer_node in trainer.nodes.items():
            if name is None:  # Root는 이미 생성됨
                continue

            self.nodes[name] = InferenceNode(self, name, trainer_node, fold_idx=fold_idx)
            node_count += 1

        print(f"✅ Created {node_count} inference node(s)")

    def get_data(self, edges):
        """edges에 따라 데이터를 가져옴"""
        data_list = []
        for node_name, var in edges:
            node = self.nodes[node_name]
            data = node.get_data(var)
            data_list.append(data)

        # 여러 노드의 결과를 concat
        if len(data_list) == 1:
            return [data_list[0]]
        else:
            # axis=1로 concat (컬럼 방향)
            concatenated = type(data_list[0]).concat(data_list, axis=1)
            return [concatenated]

    def predict(self, node_name):
        """특정 노드의 추론 결과를 반환

        Args:
            node_name: 추론할 노드 이름

        Returns:
            추론 결과 DataWrapper
        """
        if node_name not in self.nodes:
            raise ValueError(f"Node '{node_name}' not found")

        print(f"🔮 Running inference for node '{node_name}'...")
        result = self.nodes[node_name].get_data()
        print(f"✅ Inference complete")

        return result

    def predict_all(self):
        """모든 노드의 추론 결과를 dict로 반환

        Returns:
            dict: {node_name: result}
        """
        results = {}
        print(f"🔮 Running inference for all nodes...")

        for name, node in self.nodes.items():
            if name is None:  # Root 제외
                continue

            print(f"  ├─ Node '{name}'...")
            results[name] = node.get_data()

        print(f"✅ Inference complete for {len(results)} node(s)")
        return results

    def save(self, filepath):
        """Inferencer 객체를 파일로 저장

        Args:
            filepath: 저장할 파일 경로
        """
        print(f"💾 Saving Inferencer to {filepath}...")
        with open(filepath, 'wb') as f:
            pkl.dump(self, f)

        print(f"✅ Inferencer saved successfully")

    @staticmethod
    def load(filepath):
        """파일에서 Inferencer 객체를 불러옴

        Args:
            filepath: 불러올 파일 경로

        Returns:
            Inferencer: 불러온 Inferencer 객체
        """
        print(f"📂 Loading Inferencer from {filepath}...")
        with open(filepath, 'rb') as f:
            inferencer = pkl.load(f)

        print(f"✅ Inferencer loaded successfully")
        print(f"   - {len(inferencer.nodes) - 1} node(s)")

        return inferencer
