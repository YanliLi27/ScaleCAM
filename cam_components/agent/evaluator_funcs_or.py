import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
from scipy import stats
from cam_components.agent.utils.scat_plot import scatter_plot
import os


def cam_eval_step_corr(cam_algorithm, model, target_layer, target_dataset,  # required attributes
                       general_args,
                       im=None, data_max_value=None, data_min_value=None, remove_minus_flag:bool=True,
                       max_iter=None,
                       device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
    '''
    Using the logits instead of confidence, with the calculation before softmax.
    '''
    test_loader = DataLoader(target_dataset, batch_size=general_args.batch_size, shuffle=False,
                                num_workers=1, pin_memory=True)
    model.eval()
    corr_cam_matrix = []
    corr_cate_matrix = []
    corr_pred_matrix = []
    counter = 0

    for x,y in tqdm(test_loader):
        x = x.to(dtype=torch.float32).to(device)
        y = y.to(dtype=torch.float32).to(device)
        
        with cam_algorithm(model=model,
                            target_layers=target_layer,
                            importance_matrix=im,  # im -- [batch, organ_groups * channels] - [batch, 2 * N]
                            use_cuda=True,
                            groups=general_args.groups,
                            value_max=data_max_value,
                            value_min=data_min_value,
                            remove_minus_flag=remove_minus_flag,
                            out_logit=False,
                            ) as cam:

            grayscale_cam, predict_category, pred_score, nega_score = cam(input_tensor=x,
                                                                          gt=y,
                                                                          target_category=general_args.target_category_flag)
                                                                          # target_category='GT')
            # 分别是： cam图，一个预测类别：int类， 一个预测\GT的分数(已经确认为指定类分数)，其他部分的分数--可能为None

        grayscale_cam = np.array(grayscale_cam)  # # grayscale_cam -- 16 * [1, 256, 256] - batch * [1, 256, 256]
        # 3d grayscale cam -- 16 * [1, 5, 256, 256] - batch * [1, 5, 256, 256]
        # batch_size * [group_num, (z,) y, x]
        for i, single_cam in enumerate(grayscale_cam):  # 取单个进行计算和存储
            # [group_num, (z,) y, x]   # normalize to 0-1
            corr_cam_matrix.append(np.sum(single_cam))
            corr_cate_matrix.append(predict_category[i])
            corr_pred_matrix.append(pred_score[i])
            counter += 1


        if max_iter is not None:
            if counter >= max_iter:
                print('counter meet max iter')
                break
    
    print('total samples:', counter)

    # cam分数和类别的AUROC，代表的是cam正确反映分类情况的能力
    # however, not available for multiple classes
    auc = roc_auc_score(corr_cate_matrix, corr_cam_matrix)
    print('outlier rate-- AUROC of <CAM & Label>: ', auc)
    cam_son_dir = f'fold{general_args.fold_order}_cate{general_args.target_category_flag}_mt{general_args.method}'
    cam_sub_dir = f'norm{general_args.stat_maxmin_flag}_rm0{general_args.remove_minus_flag}_{general_args.im_selection_mode}{general_args.im_selection_extra}'
    save_name = os.path.join(r'.\\output\\figs', cam_son_dir, cam_sub_dir, f'or_scatter_{str(auc)[:5]}.jpg')
    scatter_plot(corr_cate_matrix, corr_cam_matrix, fit=False, save_path=save_name)
    print(f'or scatter plot saved: {save_name}')
    
    # cam分数与pred的corr，代表CAM正确反映pred的能力，也即是weight与真实重要程度的关系情况
    corr, p_value = stats.spearmanr(corr_cam_matrix, corr_pred_matrix)       
    
    print('corrlation of <CAM & Pred scores>: ', corr)
    print('p value: ', p_value)
    corr_save_name = os.path.join(r'.\\output\\figs', cam_son_dir, cam_sub_dir, f'corr_scatter_{str(corr)[:6]}_{str(p_value)[:6]}.jpg')
    scatter_plot(corr_pred_matrix, corr_cam_matrix, save_path=corr_save_name)
    print(f'corr scatter plot saved: {corr_save_name}')