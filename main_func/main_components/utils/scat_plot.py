from matplotlib import pyplot as plt
import numpy as np
import os


def scatter_plot(Garray:np.array, Parray:np.array, fit:bool=True, save_path:str='./output/figs/scatter_gt_pr.jpg'):
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))
    plt.clf()
    # set the fit curve
    linear_model=np.polyfit(Garray, Parray, 1)
    linear_model_fn=np.poly1d(linear_model)
    x_s=np.arange(0, 2)
    if fit:
        plt.plot(x_s,linear_model_fn(x_s),color="red")
    # scatter plot
    plt.scatter(Garray, Parray, s=5, c=None, marker=None, cmap=None, norm=None, alpha=0.5, linewidths=None)
    plt.savefig(save_path)
    plt.clf()

