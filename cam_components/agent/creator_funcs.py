import torch
import cv2
import os
from tqdm import tqdm
from cam_components.agent.utils.img_creator import origin_creator, cam_creator, origin_creator3d
import SimpleITK as sitk
import numpy as np
from torchvision import transforms
# for eval:
from sklearn.metrics import roc_auc_score
from scipy import stats
from cam_components.agent.utils.scat_plot import scatter_plot
from cam_components.agent.utils.eval_utils import cam_regularizer, cam_input_normalization, pred_score_calculator, text_save


def cam_creator_step(cam_algorithm, model, target_layer, dataset, num_classes:int, cam_dir:str,  # required attributes
                    # --- optional functions --- #
                    im=None, data_max_value=None, data_min_value=None, remove_minus_flag:bool=True,
                    max_iter=None, set_mode:bool=True, use_origin:bool=True,
                    batch_size:int=1, groups:int=1, target_category=None,
                    fold_order:int=0,
                    # --- eval --- #
                    eval_func:bool=False, tanh_flag:bool=False, t_max:float=0.95, t_min:float=0.05,
                    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
    if set_mode:
        in_fold_counter = 0
        model = model.to(device=device)
        model.eval()

        # --- eval --- #
        counter = 0
        assert eval_func in ['false', 'basic', 'logit', 'corr']
        if eval_func == 'corr':
            corr_cam_matrix = []
            corr_cate_matrix = []
            corr_pred_matrix = []
        elif eval_func in ['logit', 'basic']:
            increase = 0.0
            drop = 0.0

        # -------------- start cam calculation -------------- #
        for x,y in tqdm(dataset):
            if len(x.shape)==4: # x -- [batch, channel, y, x]
                origin_img = origin_creator(x, organ_groups=groups)
                # origin_img -- [batch, organ_groups, channel=3, y, x]
            elif len(x.shape)==5: # x -- [batch, channel, z, y, x]/[batch, group/channel=2, Depth, Length, Width]
                origin_img = origin_creator3d(x, organ_groups=groups)
                origin_img = origin_img.transpose(0, 1, 5, 2, 3, 4)  # because we got no way for colored 3d images
                # origin_img -- [batch, organ_groups, channel=3, z, y, x]
            x = x.to(dtype=torch.float32).to(device)
            y = y.to(dtype=torch.float32).to(device)
            
            with cam_algorithm(model=model,
                                target_layers=target_layer,
                                num_out=num_classes,
                                importance_matrix=im,  # im -- [batch, organ_groups * channels] - [batch, 2 * N]
                                use_cuda=True,
                                groups=groups,
                                value_max=data_max_value,
                                value_min=data_min_value,
                                remove_minus_flag=remove_minus_flag,
                                out_logit=False,
                                tanh_flag=tanh_flag,
                                t_max=t_max,
                                t_min=t_min
                                ) as cam:

                grayscale_cam, predict_category, pred_score, nega_score = cam(input_tensor=x,
                                                                            gt=y,
                                                                            target_category=target_category)
                # theory: grayscale_cam -- batch * (target_layer_aggregated)_array[groups, (depth), length, width]
                # proved: grayscale_cam -- 16 * [1(groups), 256, 256] - batch * [1(groups), 256, 256]

                # ---------------------------------------  cam create  --------------------------------------- #
                if not os.path.exists(cam_dir):
                    os.makedirs(cam_dir)
                for i in range(batch_size):
                    # for 2D input
                    if len(grayscale_cam[i].shape) == 3:
                        concat_img_all, output_label, cf_num = cam_creator(grayscale_cam[i], predict_category[i],\
                                                                        pred_score[i], groups, origin_img[i], \
                                                                        use_origin=use_origin)
                        # save the cam
                        str_labels = (str(y.data.cpu().numpy()[i]))[:2]
                        save_name = os.path.join(cam_dir, f'fold{fold_order}_tr{str_labels}pr{output_label}_{in_fold_counter}_cf{cf_num}.jpg')
                        in_fold_counter += 1
                        cv2.imwrite(save_name, concat_img_all)

                    # for 3D input
                    elif len(grayscale_cam[i].shape) == 4:
                        # concat_img_all, origin_img_all, output_label, cf_num = cam_creator3d(grayscale_cam[i], predict_category[i],\
                        #                                                 confidence[i], general_args.groups, origin_img[i], \
                        #                                                 use_origin=use_origin)   
                        # print(grayscale_cam[i].shape)   
                        output_label = predict_category[i]
                        cf_num = str(np.around(pred_score[i], decimals=3))
                        # save the cam
                        str_labels = (str(y.data.cpu().numpy()[i]))[:2]
                        save_name = os.path.join(cam_dir, f'fold{fold_order}_tr{str_labels}pr{output_label}_{in_fold_counter}_cf{cf_num}.nii.gz')
                        origin_save_name = save_name.replace('.nii.gz', '_ori.nii.gz')
                        in_fold_counter += 1
    
                        # TODO from [batch, organ_groups, z, y, x, channel] to [batch, organ_groups, channel, z, y, x]
                        # current we just use the second layer of input
                        for group_index in range(origin_img.shape[1]):
                            save_name = save_name.replace('.nii.gz', '_p1.nii.gz')
                            origin_save_name = origin_save_name.replace('.nii.gz', '_p1.nii.gz')
                            writter = sitk.ImageFileWriter()
                            writter.SetFileName(save_name)
                            writter.Execute(sitk.GetImageFromArray(grayscale_cam[i][group_index]))
                            writter.SetFileName(origin_save_name)
                            writter.Execute(sitk.GetImageFromArray(origin_img[i][group_index][1]))
                    else:
                        raise ValueError(f'not supported shape: {grayscale_cam[i].shape}') 
                # ---------------------------------------  cam create  --------------------------------------- #

                # --------------------------------------  cam evaluate  -------------------------------------- #
                if eval_func == 'corr':
                    grayscale_cam = np.array(grayscale_cam)  # # grayscale_cam -- 16 * [1, 256, 256] - batch * [1, 256, 256]
                    if tanh_flag and data_max_value:
                        para_k = (np.arctanh(t_max) - np.arctanh(t_min))/(data_max_value-data_min_value)
                        para_b = (np.arctanh(t_max)*data_min_value-np.arctanh(t_min)*data_max_value)/(data_min_value-data_max_value)
                        grayscale_cam = (np.arctanh(grayscale_cam) - para_b)/para_k
                    # 3d grayscale cam -- 16 * [1, 5, 256, 256] - batch * [1, 5, 256, 256]
                    # batch_size * [group_num, (z,) y, x]
                    for i, single_cam in enumerate(grayscale_cam):  # 取单个进行计算和存储
                        # [group_num, (z,) y, x]   # normalize to 0-1
                        corr_cam_matrix.append(np.sum(single_cam))
                        corr_cate_matrix.append(predict_category[i])
                        corr_pred_matrix.append(pred_score[i])
                        counter += 1
                elif eval_func in ['basic', 'logit']:
                    grayscale_cam = np.array(grayscale_cam)
                    grayscale_cam = cam_regularizer(np.array(grayscale_cam)) # -- [16, 1, 256, 256]
                    # grayscale_cam:numpy [batch, groups, length, width] from 0 to 1, x:tensor [batch, in_channel, length, width] from low to high
                    extended_cam = np.zeros(x.shape, dtype=np.float32)
                    channel_per_group = x.shape[1] // groups
                    for gc in range(groups):
                        extended_cam[:, gc*channel_per_group:(gc+1)*channel_per_group, :] = np.expand_dims(grayscale_cam[:, gc, :], axis=1)
                    # extended_cam: numpy [batch, in_channel, length, width]
                    cam_input = torch.from_numpy(extended_cam).to(device) * x
                    cam_input = cam_input_normalization(cam_input)
                    cam_pred = model(cam_input)
                    if eval_func == 'basic':
                        origin_category, single_origin_confidence = predict_category, pred_score
                        _, single_cam_confidence = pred_score_calculator(x.shape[0], cam_pred, target_category,
                                                                                    origin_pred_category=origin_category)
                        single_drop = torch.relu(torch.from_numpy(single_origin_confidence\
                                     - single_cam_confidence)).div(torch.from_numpy(single_origin_confidence) + 1e-7)
                    elif eval_func == 'logit':
                        origin_category, single_origin_confidence = y, pred_score
                        _, single_cam_confidence, single_cam_nega_scores = pred_score_calculator(x.shape[0], cam_pred, 'GT',
                                                                                            origin_pred_category=origin_category)
                        
                        single_drop = nega_score > single_cam_nega_scores  # 新的drop越大越好
                    counter += x.shape[0]
                    single_increase = single_origin_confidence < single_cam_confidence
                    increase += single_increase.sum().item()
                    drop += single_drop.sum().item()
                # --------------------------------------  cam evaluate  -------------------------------------- #

                # early stop
                if max_iter is not None:
                    if counter >= max_iter:
                        print('counter meet max iter')
                        break

        # --------------------------------------  cam evaluate  -------------------------------------- #
        if eval_func == 'corr':
            print('total samples:', counter)
            # cam分数和类别的AUROC，代表的是cam正确反映分类情况的能力
            # however, not available for multiple classes
            auc = roc_auc_score(corr_cate_matrix, corr_cam_matrix)
            print('outlier rate-- AUROC of <CAM & Label>: ', auc)
            corr_dir = cam_dir.replace('./output/cam/', './output/figs/')
            save_name = os.path.join(corr_dir, f'or_scatter_{str(auc)[:5]}.jpg')
            scatter_plot(corr_cate_matrix, corr_cam_matrix, fit=False, save_path=save_name)
            print(f'or scatter plot saved: {save_name}')
            
            # cam分数与pred的corr，代表CAM正确反映pred的能力，也即是weight与真实重要程度的关系情况
            corr, p_value = stats.spearmanr(corr_cam_matrix, corr_pred_matrix)       
            
            print('corrlation of <CAM & Pred scores>: ', corr)
            print('p value: ', p_value)
            corr_save_name = os.path.join(corr_dir, f'corr_scatter_{str(corr)[:6]}_{str(p_value)[-6:]}.jpg')
            scatter_plot(corr_pred_matrix, corr_cam_matrix, save_path=corr_save_name)
            print(f'corr scatter plot saved: {corr_save_name}')

        elif eval_func in ['basic', 'logit']:
            print('total samples:', counter)
            avg_increase = increase/counter
            avg_drop = drop/counter

            print('increase:', avg_increase)
            print('avg_drop:', avg_drop)
            eval_borl_save_name = os.path.join(corr_dir, f'eval_with_{eval_func}.txt')
            text_save(eval_borl_save_name, avg_increase, avg_drop, counter)
        # --------------------------------------  cam evaluate  -------------------------------------- #

    else: 
        print('input is not a dataset, but something else')
        model = model.to(device=device)
        model.eval()
        trans_val = transforms.Compose([transforms.ToTensor()])
        x = trans_val(dataset) # from [C, L, D] to [1, C, L, D]
        if len(x.shape) == 3:
            x = torch.unsqueeze(x, dim=0)
        origin_img = origin_creator(x, organ_groups=groups)
        x = x.to(dtype=torch.float32).to(device)
        with cam_algorithm(model=model,
                                target_layers=target_layer,
                                importance_matrix=im,  # im -- [batch, organ_groups * channels] - [batch, 2 * N]
                                use_cuda=True,
                                groups=groups,
                                value_max=data_max_value,
                                value_min=data_min_value,
                                remove_minus_flag=remove_minus_flag
                                ) as cam:
                grayscale_cam, predict_category, pred_score, nega_score = cam(input_tensor=x, 
                                                                target_category=target_category)
                for i in range(batch_size):
                    concat_img_all, output_label, cf_num = cam_creator(grayscale_cam[i], predict_category[i], pred_score[i], groups, origin_img[i])
                    if not os.path.exists(cam_dir):
                        os.makedirs(cam_dir)
                    save_name = os.path.join(cam_dir, f'example.jpg')
                    in_fold_counter += 1
                    cv2.imwrite(save_name, concat_img_all)