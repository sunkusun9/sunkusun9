import re
import pandas as pd

class Metric:
    def __init__(
        self, name, e, target_edge, output_var, metric_func, include_train = False
    ):
        self.e = e
        self.name = name
        self.target_edge = target_edge
        self.output_var = output_var
        self.include_train = include_train
        self.metric_func = metric_func

    def get_metric(self, idx, node):
        if self.include_train:
            iterator = zip(
                self.e.get_node_output(idx, self.target_edge[0], self.target_edge[1]),
                self.e.get_node_output(idx, node, self.output_var)
            )
            for no, ((true_train, true_valid), (prd_train, prd_valid)) in enumerate(iterator):
                result_train = self.metric_func(true_train[0].data, prd_train[0].data)
                result_valid = self.metric_func(true_valid.data, prd_valid.data)
                if true_train[1] is not None:
                    result_sub = {
                        (idx, 'train', f'train_{no}'): result_train,
                        (idx, 'train', f'valid_{no}'): self.metric_func(true_train[1].data, prd_train[1].data),
                        (idx, 'valid', ''): result_valid
                    }
                else:
                    result_sub = {
                        (idx, 'train'): result_train, (idx, 'valid'): result_valid
                    }
        else:
            iterator = zip(
                self.e.get_node_valid_output(idx, self.target_edge[0], self.target_edge[1]),
                self.e.get_node_valid_output(idx, node, self.output_var)
            )
            for true_valid, prd_valid in iterator:
                result_valid = self.metric_func(true_valid.data, prd_valid.data)
                result_sub = {idx: result_valid}

        return pd.Series(result_sub)

            