from abc import ABC, abstractmethod


class BaseAnalyzer(ABC):
    requires_data = False  # X, y 데이터 필요 여부 (서브클래스에서 오버라이드)

    def __init__(self, experimenter):
        self.experimenter = experimenter
        self.results = {}

    def _start(self, node):
        self.results[node] = []

    @abstractmethod
    def _analyze(self, node, idx):
        pass

    def _end(self, node):
        pass

    def get_result(self, node):
        return self.results.get(node)

    def get_results(self, nodes=None):
        if nodes is None:
            return self.results
        return {k: v for k, v in self.results.items() if k in nodes}
