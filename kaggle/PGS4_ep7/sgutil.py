from IPython.display import Image
import os
import matplotlib.pyplot as plt
import pickle as pkl

class SGCache:
    def __init__(self, img_path, result_path):
        """
        Parameters:
            img_path: str
                이미지가 저장될 경로
            result_path: str
                수행결과가 저장될 경로
        """
        self.img_path = img_path
        self.result_path = result_path
        
    def cache_fig(self, img_name, img_func, nrow=1, ncol=1, figsize=(8, 3), redraw=False):
        """
        기존에 차트를 출력했던 결과가 있으면 기존 결과를 출력하고, 
        없다면 전달된 출력함수를 사용하여 차트를 출력합니다.
        
        Parameters:
            img_name: str
                이미지 파일명
            img_func: function name(axes)
                이미지 출력 함수
            nrow: int
                행의 수
            ncol: int
                열의 수
            figsize: tuple(width, height)
                차트 크기
            redraw: boolean
                다시 그리기 여부, True 새로 이미지를 생성하여 출력, False: 기생성된 이미지로 그리기
        Example
        >>> import seaborn as sns
        >>> sc = SGCache('img', 'result')
        >>> sc.cache_fig('hist', lambda x: sns.histplot(some_values), nrow=1, ncol=1, figsize=(8, 3), redraw=False)
        """
        img_file_name = os.path.join(self.img_path, img_name + '.png')
        if not os.path.exists(img_file_name) or redraw:
            fig, axes = plt.subplots(nrow, ncol, figsize=figsize)
            if nrow * ncol > 1:
                img_func(axes.ravel())
            else:
                img_func(axes)
            plt.tight_layout()
            plt.savefig(img_file_name)
            plt.show()
        else:
            display(Image(filename=img_file_name))
    
    def cache_result(self, result_name, result_func, rerun=False):
        """
        기존에 차트를 출력했던 결과가 있으면 기존 결과를 반환하고,
        없다면 전달된 수행함수를 실행하고 결과를 뽑아 반환합니다.
        
        Parameters:
            result_name: str
                결과명
            result_func: function name(axes)
                결과 생성 함수
            rerun: boolean
                재실행, True 새로 수행하여 결과 반환, False: 기존 결과가 있다면 반환
        
        Example
        >>> import pandas as pd
        >>> ... 
        >>> sc = SGCache('img', 'result')
        >>> s_sum_by_grp = sc.cache_result('sum_by_grp', lambda : df.groupby('grp')['value'].sum() , rerun=False)
        """
        result_file_name = os.path.join(self.result_path, result_name + '.pkl')
        if not os.path.exists(result_file_name) or rerun:
            result = result_func()
            if result is None:
                return
            with open(result_file_name, 'wb') as f:
                pkl.dump(result, f)
        else:
            with open(result_file_name, 'rb') as f:
                result = pkl.load(f)
        return result
