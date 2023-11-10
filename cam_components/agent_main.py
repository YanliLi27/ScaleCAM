import os
import cv2
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from typing import Union
import numpy as np
# for 3D input
import SimpleITK as sitk
# analyzer
from cam_components.agent.im_funcs import im_save
from cam_components.agent.analyzer_utils import stat_calculator
# creator
from cam_components.agent.target_cam_calculation import target_cam_selection
from cam_components.agent.im_funcs import im_reader, maxmin_reader
from cam_components.agent.img_creator import origin_creator, cam_creator, origin_creator3d
# for evaluation
from sklearn.metrics import roc_auc_score
from scipy import stats
from cam_components.agent.scat_plot import scatter_plot
from cam_components.agent.eval_utils import cam_regularizer, cam_input_normalization, pred_score_calculator, text_save
from typing import Union
from scipy.special import softmax
# cam methods
from cam_components.methods import GradCAM_A, GradCAM_P, FullCAM_A, FullCAM_P, GradCAMPP_A, GradCAMPP_P, XGradCAM_A, XGradCAM_P


class CAMAgent:
    def __init__(self,
                # ---------------- model and dataset -------------#
                model, target_layer, dataset:Union[DataLoader, np.array],  
                num_out_channel:int=512, num_classes:int=2,  
                groups:int=1, fold_order:int=0, ram:bool=False,
                # ---------------- model and dataset -------------#
                cam_method:str='gradcam', im_dir:str='./output/im/taskname', cam_dir:str='./output/cam/taskname',
                # cam method and im paths and cam output
                batch_size:int=1, target_category:Union[None, str, int, list]=1,  # info of the running process
                maxmin_flag:bool=False, remove_minus_flag:bool=True, # creator
                im_selection_mode:str='all', im_selection_extra:float=0.05, # importance matrices attributes
                max_iter=None,  # early stop
                randomization:bool=False, random_severity:int=1,  # model randomization for sanity check
                use_pred:bool=False,  # improve effeciency
                ) -> None:
        

        # cam info
        cam_method_zoo = {"gradcam": [GradCAM_A, GradCAM_P], 
                        "fullcam": [FullCAM_A, FullCAM_P],
                        "gradcampp":[GradCAMPP_A, GradCAMPP_P],
                        "xgradcam":[XGradCAM_A, XGradCAM_P]}
        self.cam_method_name = cam_method
        self.cam_method = cam_method_zoo[cam_method]

        # model info
        self.model = model
        self.target_layer = target_layer
        self.num_out_channel = num_out_channel
        self.batch_size = batch_size

        # dataset info
        self.num_classes = num_classes
        self.dataset = dataset
        self.ram = ram
        if self.ram:
            print('please notice, for regression tasks, the target categories is necessary, for both analyzer and creator.\
                  if no predefined category, the default is 0.')

        assert (target_category in ['GT', None] or type(target_category) == int or type(target_category)==list)
        self.target_category = target_category  # targeted category

        self.groups = groups  # group convolution
        self.fold_order = fold_order  # if use cross-validation

        # Extra function of the CAM creator
        self.maxmin_flag = maxmin_flag
        self.remove_minus_flag = remove_minus_flag
        assert im_selection_mode in ['max', 'top', 'diff_top', 'freq', 'index', 'all']
        self.im_selection_mode = im_selection_mode
        self.im_selection_extra = im_selection_extra
        self.use_pred = use_pred
        
        # early stop
        if max_iter is not None:
            self.max_iter = max_iter
        else:
            self.max_iter = None

        # model randomization for sanity check
        self.randomization = randomization
        assert random_severity in [0, 1, 2, 3, 4]
        self.random_severity = random_severity

        # output info
        self.im_dir = im_dir
        self.cam_dir = cam_dir
        

        print(f'------------------------------ initialized ------------------------------')
    

    def _im_finder(self, target_category):
        if self.use_pred:
            im_name = f'All_fold{self.fold_order}_im_cateNone_{self.cam_method_name}.csv' 
        else:
            im_name = f'All_fold{self.fold_order}_im_cate{target_category}_{self.cam_method_name}.csv' 
        im_path = os.path.join(self.im_dir, im_name)
        return im_path


    def _cam_finder(self, target_category):
        cam_son_dir = f'fold{self.fold_order}_cate{target_category}_mt{self.cam_method_name}'
        cam_sub_dir = f'norm{self.maxmin_flag}_rm0{self.remove_minus_flag}_{self.im_selection_mode}{self.im_selection_extra}'
        cam_dir = os.path.join(self.cam_dir, cam_son_dir, cam_sub_dir)
        if not os.path.exists(cam_dir):
            os.makedirs(cam_dir)
        return cam_dir


    def analyzer_main(self, confidence_weight_flag:bool=False):
        if type(self.target_category)==list:
            for tc in self.target_category:
                im_path = self._im_finder(tc)
                print('im_path: {}'.format(im_path))
                if not os.path.isfile(im_path):  # only when the file dosen't exist -- because some loop would be repeated in experiments
                    print(f'--------- creating IMs for target {tc} ---------')
                    im_overall, im_target, im_diff, cam_grad_max_matrix, cam_grad_min_matrix \
                        = self._cam_stats_step(self.target_layer, # for the cam setting
                                        target_category=tc,
                                        confidence_weight_flag=confidence_weight_flag
                                        )
                    im_save(im_overall, im_target, im_diff, cam_grad_max_matrix, cam_grad_min_matrix,
                            im_path)
        else:
            im_path = self._im_finder(self.target_category)
            print('im_path: {}'.format(im_path))
            if not os.path.isfile(im_path):  # only when the file dosen't exist -- because some loop would be repeated in experiments
                if self.target_category == None or self.target_category=='GT':
                    print(f'--------- creating IMs for all categories ---------')
                else:
                    print(f'--------- creating IMs for target {self.target_category} ---------')
                im_overall, im_target, im_diff, cam_grad_max_matrix, cam_grad_min_matrix \
                    = self._cam_stats_step(self.target_layer, # for the cam setting
                                        target_category=self.target_category,
                                        confidence_weight_flag=confidence_weight_flag
                                        )
                im_save(im_overall, im_target, im_diff, cam_grad_max_matrix, cam_grad_min_matrix,
                        im_path)


    def _cam_stats_step(self, target_layers, # for the cam setting
                        target_category:Union[None, int, str, list]=1, # the target category for cam
                        confidence_weight_flag:bool=False,  # if prefer to weight the importance with confidence
                        device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
        in_fold_counter = 0
        in_fold_target_counter = np.zeros([self.num_classes], dtype=np.int16)
        # TODO improve the num_out_channel calculation
        im_overall = np.zeros([self.num_out_channel], dtype=np.float32)  # 400 = number of channels per group
        im_target = np.zeros([self.num_classes, self.num_out_channel], dtype=np.float32)
        im_diff = np.zeros([self.num_classes, self.num_out_channel], dtype=np.float32)
        # cammax & cammin for overall max-min normalization
        cam_grad_max_matrix = []
        cam_grad_min_matrix = []
        
        model = self.model.to(device=device)
        model.eval()
        cuda_flag = True if device=='cuda' else False

        for x,y in tqdm(self.dataset):
            x = x.to(dtype=torch.float32).to(device)
            y = y.to(dtype=torch.float32).to(device)
            with self.cam_method[0](model=model,
                                    ram=self.ram,
                                    target_layers=target_layers,
                                    use_cuda=cuda_flag) as cam:
                grayscale_cam, predict_category, confidence, cam_grad_max_value, cam_grad_min_value\
                                                                                        = cam(input_tensor=x, 
                                                                                            target_category=target_category,
                                                                                            )
                # cam_single_max_value - [batch, 1*value]已经展平 --[batch]
                cam_grad_max_matrix.extend(cam_grad_max_value)
                cam_grad_min_matrix.extend(cam_grad_min_value)

                # proved: grayscale_cam - 1* [16, 512]
                if type(grayscale_cam)==list:
                    grayscale_cam = grayscale_cam[0]  # [1, all_channel] remove the target layers
                # grayscale_cam - [16, 512]
                for i in range(self.batch_size): # [all_channel]
                    single_grayscale_cam, single_predict_category, single_confidence = grayscale_cam[i], predict_category[i], confidence[i]
                    if confidence_weight_flag:
                        single_grayscale_cam = single_grayscale_cam * single_confidence
                        single_max_reviser = single_confidence
                    else:
                        single_max_reviser = 1

                    if self.ram:
                        if single_predict_category == y.data.cpu().numpy()[i][target_category]:
                            # 添加总体IM
                            im_overall = im_overall + single_grayscale_cam
                            in_fold_counter += 1
                            # 添加对应类的IM
                            
                            im_target[single_predict_category] = im_target[single_predict_category] + single_grayscale_cam
                            in_fold_target_counter[single_predict_category] += single_max_reviser
                    else:
                        if single_predict_category == y.data.cpu().numpy()[i]:  # 只叠加正确分类的部分
                            # 添加总体IM
                            im_overall = im_overall + single_grayscale_cam
                            in_fold_counter += 1
                            # 添加对应类的IM
                            
                            im_target[single_predict_category] = im_target[single_predict_category] + single_grayscale_cam
                            in_fold_target_counter[single_predict_category] += single_max_reviser
        # im_target - [num_classes, num_features]
        im_overall = im_overall / in_fold_counter
        im_target = im_target / in_fold_target_counter[:, None]
        # TODO figure out how it works for None input
        for i in range(self.num_classes):
            im_diff[i] = im_target[i, :] - im_overall

        # im_overall [num_out_channel]
        # im_target/im_diff [num_classes, num_out_channel]
        # 此处im不分group因为不同group的feature在heatmap上就应该不同，在重要性上的差异也应该保留
        # 而max min不分group或者类别因为需要全局统一尺度，无论group或者batch或者category
        # 计算分位数
        cam_grad_max_matrix = np.array(cam_grad_max_matrix)
        cam_grad_min_matrix = np.array(cam_grad_min_matrix)
        cam_grad_max_matrix, cam_grad_min_matrix = stat_calculator(cam_grad_max_matrix, cam_grad_min_matrix)
        return im_overall, im_target, im_diff, cam_grad_max_matrix, cam_grad_min_matrix


    def creator_main(self, creator_target_category:Union[None, str, int, list]='Default',
                    # if wanted to target a category while the analyzer using None
                    eval_act:Union[bool, str]=False, mm_ratio:float=1.5, 
                    cam_save:bool=True, use_origin:bool=True,
                    cluster:Union[None, str, list]=None, cluster_start:int=0,
                    tanh_flag:bool=False, backup_flag:bool=False, img_compress:bool=False):
        '''
        mm_ratio for better visuaization
        use_origin for overlay/or not
        '''
        if creator_target_category=='Default':
            creator_target_category = self.target_category
             # the load im for cam creator
            # this could be None, int, list, when self.target_category == None
            # None: get the prediction-related CAMs, while the target_category for IM creation is None
            # int&list: get the CAM for certain category, while the target_category for IM creation is None

        # step 3. pred step
        if type(self.target_category)==list:
            self._cam_creator_step_width(self.target_layer, mm_ratio, use_origin, backup_flag, tanh_flag,
                                        cluster, cluster_start, compress=img_compress,
                                        creator_target_category=creator_target_category)
        else:
            self._cam_creator_step_depth(self.target_layer, mm_ratio, use_origin, cam_save=cam_save, 
                                         backup_flag=backup_flag, eval_func=eval_act, tanh_flag=tanh_flag,
                                         compress=img_compress,
                                         creator_target_category=creator_target_category)
        

    def _get_importances(self, im_path, mm_ratio):
        if os.path.exists(im_path) and self.im_selection_mode != 'all':
            print('loading importance matrix with mode:{}'.format(self.im_selection_mode))
            im = im_reader(im_path, self.im_selection_mode)
            # im [num_classes, num_features]
            im = target_cam_selection(im, mode=self.im_selection_mode, extra=self.im_selection_extra)
            # im [num_classes, num_features]
        else:
            im = None
        
        # step 2. max-min normalization - or not
        if im_path and self.maxmin_flag:
            data_max_value, data_min_value = maxmin_reader(im_path, self.target_category)
            data_max_value, data_min_value = data_max_value * mm_ratio, data_min_value * mm_ratio
        else:
            data_max_value, data_min_value = None, None
        return im, data_max_value, data_min_value


    def _cam_creator_step_width(self,  target_layer, # required attributes
                                # --- optional functions --- #
                                mm_ratio:float=1.5,
                                use_origin:bool=True,
                                backup_flag:bool=False,
                                # --- eval --- #
                                tanh_flag:bool=False, 
                                cluster:Union[None, str, list]=None, cluster_start:int=0,
                                t_max:float=0.95, t_min:float=0.05,
                                # --- img function --- #
                                compress:bool=False,
                                creator_target_category:Union[None, str, int, list]='Default',
                                device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
        
        # get importance matrices for each category
        im_box = {}
        max_box = {}
        min_box = {}
        if creator_target_category=='Default':
            creator_target_category = self.target_category
            print('use default and something goes wrong with previous assertion')
        creator_tc = creator_target_category if self.target_category==None else self.target_category
        for tc in creator_tc:
            assert type(tc)==int
            im_path = self._im_finder(tc)
            im_box[str(tc)], max_box[str(tc)], min_box[str(tc)] = self._get_importances(im_path, mm_ratio)
        # cam dir for multi category
        cam_dir = self._cam_finder('Multiclass')

        # for cam calculation
        in_fold_counter = 0
        model = self.model.to(device=device)
        model.eval()
        cuda_flag = True if device=='cuda' else False

        # -------------- start cam calculation -------------- #
        for x,y in tqdm(self.dataset):
            if cluster=='signal':  # x -- [batch, width, lengths]
                origin_img = x
            elif len(x.shape)==4: # x -- [batch, channel, y, x]
                origin_img = origin_creator(x, organ_groups=self.groups, compress=compress)
                # origin_img -- [batch, organ_groups, channel=3, y, x]
            elif len(x.shape)==5: # x -- [batch, channel, z, y, x]/[batch, group/channel=2, Depth, Length, Width]
                origin_img = origin_creator3d(x, organ_groups=self.groups)
            else:
                raise ValueError('unsupported input: not signal(batch, width, lengths), and not 2D, 2.5D or 3D images')
            if len(origin_img.shape)>5:
                origin_img = origin_img.transpose(0, 1, 5, 2, 3, 4)  # because we got no way for colored 3d images
                # origin_img -- [batch, organ_groups, channel=3, z, y, x]
            x = x.to(dtype=torch.float32).to(device)
            y = y.to(dtype=torch.float32).to(device)

            tc_cam = []
            tc_pred_category = []
            tc_score = []
            tc_nega_score = []

            for tc in creator_tc:
                with self.cam_method[1](model=model,
                                        target_layers=target_layer,
                                        ram=self.ram,
                                        importance_matrix=im_box[str(tc)],  # im -- [batch, organ_groups * channels] - [batch, 2 * N]
                                        use_cuda=cuda_flag,
                                        groups=self.groups,
                                        value_max=max_box[str(tc)],
                                        value_min=min_box[str(tc)],
                                        remove_minus_flag=self.remove_minus_flag,
                                        out_logit=False,
                                        tanh_flag=tanh_flag,
                                        t_max=t_max,
                                        t_min=t_min
                                        ) as cam:

                    grayscale_cam, predict_category, pred_score, nega_score = cam(input_tensor=x,
                                                                                gt=y,
                                                                                target_category=tc)
                    # theory: grayscale_cam -- batch * (target_layer_aggregated)_array[groups, (depth), length, width]
                    # proved: grayscale_cam -- 16 * [1(groups), 256, 256] - batch * [1(groups), 256, 256]

                tc_cam.append(grayscale_cam)  # tc_cam: tc_len* batch* (target_layer_aggregated)_array[groups, (depth), length, width]
                # tc_cam: [5(tc) * [16 * [1(groups), 256, 256]]] / [5(tc) * [16 * [1(groups), depth, 256, 256]]]
                tc_pred_category.append(predict_category)
                # tc_pred_category: [tc_len*1]
                tc_score.append(pred_score)
                # tc_score: [tc_len*1]
                tc_nega_score.append(nega_score)
                # tc_nega_score: [tc_len*1]

            # ---------------------------------------  cam create  --------------------------------------- #
            if not os.path.exists(cam_dir):
                os.makedirs(cam_dir)
            if backup_flag:
                backup_dir = cam_dir.replace('./output/cam/', './output/backup/')
                if not os.path.exists(backup_dir):
                    os.makedirs(backup_dir)
            # step
            # cluster for the multi-target CAM generation -- e.g. SYN/TSY/BME
            if type(cluster)==list:
                assert np.sum(cluster)<=len(creator_tc)  # make sure the length equal
                assert len(cluster)<=3  # only 3 channels for RGB
                # tc_cam for cluster -- [tc]
                tc_cam = np.asarray(tc_cam)
                if len(tc_cam.shape)==5:
                    tc_cam = np.transpose(np.asarray(tc_cam), (1, 2, 0, 3, 4)) 
                    # tc_cam  from [tc, batch, groups, width, length] to [batch, groups, tc, width, length]
                    B, G, _, W, L = tc_cam.shape
                    clustered_cam = np.zeros((B,G,3,W,L))
                elif len(tc_cam.shape)==6:
                    tc_cam = np.transpose(np.asarray(tc_cam), (1, 2, 0, 3, 4, 5)) 
                    # tc_cam  from [tc, batch, groups, width, length] to [batch, groups, tc, depth, width, length]
                    B, G, _, D, W, L = tc_cam.shape
                    clustered_cam = np.zeros((B,G,3,D,W,L))
                else:
                    raise ValueError(f'The shape for width-first CAM of {tc_cam.shape} is not supported')
                cluster_counter = cluster_start
                for i in range(len(cluster)):
                    clustered_cam[:, :, i] = np.sum(tc_cam[:, :, cluster_counter:cluster_counter+cluster[i]], axis=2)
                    cluster_counter+=cluster[i]
                # clustered_cam [batch, groups, 3, (depth),width, length]

                # missing dim because of 2.5D
                if len(clustered_cam.shape)<len(origin_img.shape):
                    # from [batch, groups, 3, width, length] to [batch, groups, 3, depth, width, length]
                    clustered_cam = np.expand_dims(clustered_cam, axis=3)
                    # [2, 2, 3, 1, 512, 512]
                    
                for i in range(B):
                    in_fold_counter += 1
                    for j in range(G):
                        if len(clustered_cam.shape)==5:
                            concat_img_all, output_label, cf_num = cam_creator(clustered_cam[i], tc_pred_category[i],\
                                                                            tc_score[i], self.groups, origin_img[i], \
                                                                            use_origin=use_origin)
                            # save the cam
                            save_name = os.path.join(cam_dir, f'fold{self.fold_order}_pr{output_label}_target{creator_tc[i]}_{in_fold_counter}_cf{cf_num}.jpg')
                            cv2.imwrite(save_name, concat_img_all)
                            # save the backup npy for further calculation
                            if backup_flag:
                                backup_name = os.path.join(backup_dir, f'fold{self.fold_order}_pr{output_label}_target{creator_tc[i]}_{in_fold_counter}_cf{cf_num}.npy')
                                np.save(backup_name, np.asarray({'img':origin_img[i],'cam':clustered_cam[i]}))
                                # grayscale_cam [-batch- * [1(groups), 256, 256]]
                                # origin_img [-batch-, organ_groups, channel=3, y, x]
                        elif len(clustered_cam.shape)==6:
                            output_label = tc_pred_category[i]
                            cf_num = str(np.around(tc_score[i], decimals=3))
                            # save the cam
                            save_name = os.path.join(cam_dir, f'fold{self.fold_order}_{in_fold_counter}_pr{output_label}_cf{cf_num}.nii.gz')
                            origin_save_name = save_name.replace('.nii.gz', '_ori.nii.gz')
                            
                            for tc_index in range(len(cluster)):
                                neo_save_name = save_name.replace('.nii.gz', '_gro{}_clu{}.nii.gz'.format(j, tc_index))
                                neo_origin_save_name = origin_save_name.replace('.nii.gz', '_gro{}.nii.gz'.format(j))
                                writter = sitk.ImageFileWriter()
                                writter.SetFileName(neo_save_name)
                                writter.Execute(sitk.GetImageFromArray(clustered_cam[i][j][tc_index]))
                                if os.path.isfile(neo_origin_save_name):
                                    continue
                                else:
                                    writter.SetFileName(neo_origin_save_name)
                                    # [batch, organ_groups, z, y, x, channel] to [batch, organ_groups, z, y, x]
                                    # TODO currently we just use the second layer of input:
                                    writter.Execute(sitk.GetImageFromArray(origin_img[i][j][1]))
                        else:
                            raise ValueError(f'not supported shape: {clustered_cam.shape}') 
   

            elif type(cluster)=='signal':
                # tc_cam for signals -- [tc * [batch * [groups, width(1), length(3000)]]]
                tc_cam = np.transpose(np.asarray(tc_cam), (1, 2, 3, 0, 4))  
                # from [tc, batch, groups, width, length] to [batch, groups, width, tc, length]
                for i in range(self.batch_size):
                    in_fold_counter += 1
                    save_name = os.path.join(cam_dir, f'fold{self.fold_order}_{in_fold_counter}_pr{output_label}_cf{cf_num}_.npy')
                    np.save(save_name, np.asarray({'img':origin_img[i],'cam':tc_cam[i]}))            

            else:  # get one by one
                tc_cam = np.asarray(tc_cam)
                if len(tc_cam.shape)==5:
                    tc_cam = np.transpose(np.asarray(tc_cam), (1, 0, 2, 3, 4)) 
                    # tc_cam  from [tc, batch, groups, width, length] to [batch, groups, tc, width, length]
                    B, TC, G, W, L = tc_cam.shape
                elif len(tc_cam.shape)==6:
                    tc_cam = np.transpose(np.asarray(tc_cam), (1, 0, 2, 3, 4, 5)) 
                    # tc_cam  from [tc, batch, groups, width, length] to [batch, tc, groups, depth, width, length]
                    B, TC, G, D, W, L = tc_cam.shape
                else:
                    raise ValueError(f'The shape for width-first CAM of {tc_cam.shape} is not supported')
                # missing dim because of 2.5D
                if len(tc_cam.shape)<len(origin_img.shape):
                    # from [batch, tc, groups, width, length] to [batch, tc, groups, depth, width, length]
                    tc_cam = np.expand_dims(tc_cam, axis=3)
                    # [2, 15, 2, 1, 512, 512]

                # tc_cam [B(batch), TC(category), G(groups), (depth), W, L] 5/6
                for i in range(B):
                    in_fold_counter += 1
                    for tc in range(TC):
                        if len(tc_cam.shape)==5:
                            # for 2D input
                            concat_img_all, output_label, cf_num = cam_creator(tc_cam[i][tc], tc_pred_category[i],\
                                                                            tc_score[i], self.groups, origin_img[i], \
                                                                            use_origin=use_origin)
                            # save the cam
                            save_name = os.path.join(cam_dir, f'fold{self.fold_order}_{in_fold_counter}_pr{output_label}_target{tc}_cf{cf_num}.jpg')
                            cv2.imwrite(save_name, concat_img_all)

                            # save the backup npy for further calculation
                            if backup_flag:
                                backup_name = os.path.join(backup_dir, f'fold{self.fold_order}_{in_fold_counter}_pr{output_label}_target{tc}_cf{cf_num}.npy')
                                np.save(backup_name, np.asarray({'img':origin_img[i],'cam':tc_cam[i][tc]}))
                                # grayscale_cam [-batch- * [1(groups), 256, 256]]
                                # origin_img [-batch-, organ_groups, channel=3, y, x]
                        
                        elif len(tc_cam.shape)==6:
                            # for 3D input
                            output_label = tc_pred_category[i]
                            cf_num = str(np.around(tc_score[i], decimals=3))
                            # save the cam
                            save_name = os.path.join(cam_dir, f'fold{self.fold_order}_{in_fold_counter}_pr{output_label}_cf{cf_num}.nii.gz')
                            origin_save_name = save_name.replace('.nii.gz', '_ori.nii.gz')
                            
                            for group_index in range(G):
                                neo_save_name = save_name.replace('.nii.gz', '_gro{}_site{}.nii.gz'.format(group_index, tc))
                                neo_origin_save_name = origin_save_name.replace('.nii.gz', '_gro{}.nii.gz'.format(group_index))
                                writter = sitk.ImageFileWriter()
                                writter.SetFileName(neo_save_name)
                                writter.Execute(sitk.GetImageFromArray(tc_cam[i][tc][group_index]))
                                if os.path.isfile(neo_origin_save_name):
                                    continue
                                else:
                                    writter.SetFileName(neo_origin_save_name)
                                    # [batch, organ_groups, z, y, x, channel] to [batch, organ_groups, z, y, x]
                                    # TODO currently we just use the second layer of input:
                                    writter.Execute(sitk.GetImageFromArray(origin_img[i][group_index][1]))
                        else:
                            raise ValueError(f'not supported shape: {tc_cam.shape}') 


    def _cam_creator_step_depth(self,  target_layer, # required attributes
                                # --- optional functions --- #
                                mm_ratio:float=1.5,
                                use_origin:bool=True,
                                cam_save:bool=True,
                                backup_flag:bool=False,
                                # --- eval --- #
                                eval_func:Union[bool, str]=False, tanh_flag:bool=False, t_max:float=0.95, t_min:float=0.05,
                                # --- img function --- #
                                compress:bool=False,
                                creator_target_category:Union[None, str, int, list]='Default',
                                device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
        # im related
        im_path = self._im_finder(self.target_category)
        im, data_max_value, data_min_value = self._get_importances(im_path, mm_ratio)
        # cam output path
        if creator_target_category=='Default':
            creator_target_category = self.target_category
            print('use default and something goes wrong with previous assertion')
        creator_tc = creator_target_category if self.target_category==None else self.target_category
        cam_dir = self._cam_finder(creator_tc)

        # for cam calculation
        in_fold_counter = 0
        model = self.model.to(device=device)
        model.eval()
        cuda_flag = True if device=='cuda' else False

        # --- eval --- #
        counter = 0
        assert eval_func in [False, 'false', 'basic', 'logit', 'corr', 'corr_logit']
        logit_flag = False
        if eval_func == 'corr':
            corr_cam_matrix = []
            corr_cate_matrix = []
            corr_pred_matrix = []
        elif eval_func == 'corr_logit':
            corr_cam_matrix = []
            corr_cate_matrix = []
            corr_pred_matrix = []
            logit_flag = True
        elif eval_func in ['logit', 'basic']:
            increase = 0.0
            drop = 0.0
            acc_ori = []
            acc_cam = []
            acc_gt = []
            if eval_func in ['logit']:
                logit_flag = True

        # --- eval --- #

        # -------------- start cam calculation -------------- #
        for x,y in tqdm(self.dataset):
            if len(x.shape)==4: # x -- [batch, channel, y, x]
                origin_img = origin_creator(x, organ_groups=self.groups, compress=compress)
                # origin_img -- [batch, organ_groups, channel=3, y, x]
            elif len(x.shape)==5: # x -- [batch, channel, z, y, x]/[batch, group/channel=2, Depth, Length, Width]
                origin_img = origin_creator3d(x, organ_groups=self.groups)
            else:
                raise ValueError('unsupported input: not signal(batch, width, lengths), and not 2D, 2.5D or 3D images')
            if len(origin_img.shape)>5:
                origin_img = origin_img.transpose(0, 1, 5, 2, 3, 4)  # because we got no way for colored 3d images
            x = x.to(dtype=torch.float32).to(device)
            y = y.to(dtype=torch.float32).to(device)
            
            with self.cam_method[1](model=model,
                                    target_layers=target_layer,
                                    ram=self.ram,
                                    importance_matrix=im,  # im -- [batch, organ_groups * channels] - [batch, 2 * N]
                                    use_cuda=cuda_flag,
                                    groups=self.groups,
                                    value_max=data_max_value,
                                    value_min=data_min_value,
                                    remove_minus_flag=self.remove_minus_flag,
                                    out_logit=logit_flag,
                                    tanh_flag=tanh_flag,
                                    t_max=t_max,
                                    t_min=t_min
                                    ) as cam:

                grayscale_cam, predict_category, pred_score, nega_score = cam(input_tensor=x,
                                                                            gt=y,
                                                                            target_category=creator_tc)
                # theory: grayscale_cam -- batch * (target_layer_aggregated)_array[groups, (depth), length, width]
                # proved: grayscale_cam -- 16 * [1(groups), 256, 256] - batch * [1(groups), 256, 256]
                # pred_score -- while logit: logit, while basic: softmaxed logit
    
                # ---------------------------------------  cam create  --------------------------------------- #
                if not os.path.exists(cam_dir):
                    os.makedirs(cam_dir)
                if backup_flag:
                    backup_dir = cam_dir.replace('./output/cam/', './output/backup/')
                    if not os.path.exists(backup_dir):
                        os.makedirs(backup_dir)
                # step
                for i in range(self.batch_size):
                    # for 2D input
                    if len(grayscale_cam[i].shape) == 3:
                        concat_img_all, output_label, cf_num = cam_creator(grayscale_cam[i], predict_category[i],\
                                                                        pred_score[i], self.groups, origin_img[i], \
                                                                        use_origin=use_origin)
                        # save the cam
                        str_labels = (str(y.data.cpu().numpy()[i]))[:2]
                        save_name = os.path.join(cam_dir, f'fold{self.fold_order}_{in_fold_counter}_tr{str_labels}pr{output_label}_cf{cf_num}.jpg')
                        in_fold_counter += 1
                        if cam_save:
                            cv2.imwrite(save_name, concat_img_all)

                        # save the backup npy for further calculation
                        if backup_flag:
                            backup_name = os.path.join(backup_dir, f'fold{self.fold_order}_{in_fold_counter}_tr{str_labels}pr{output_label}_cf{cf_num}.npy')
                            np.save(backup_name, np.asarray({'img':origin_img[i],'cam':grayscale_cam[i]}))
                            # grayscale_cam [-batch- * [1(groups), 256, 256]]
                            # origin_img [-batch-, organ_groups, channel=3, y, x]

                    # for 3D input
                    elif len(grayscale_cam[i].shape) == 4:  
                        # (target_layer_aggregated)_array[groups, (depth), length, width]
                        output_label = predict_category[i]
                        cf_num = str(np.around(pred_score[i], decimals=3))
                        # save the cam
                        str_labels = (str(y.data.cpu().numpy()[i]))[:2]
                        save_name = os.path.join(cam_dir, f'fold{self.fold_order}_{in_fold_counter}_tr{str_labels}pr{output_label}_cf{cf_num}.nii.gz')
                        origin_save_name = save_name.replace('.nii.gz', '_ori.nii.gz')
                        in_fold_counter += 1
                        if cam_save:
                            for group_index in range(origin_img.shape[1]):
                                save_name = save_name.replace('.nii.gz', '_p{}.nii.gz'.format(group_index))
                                origin_save_name = origin_save_name.replace('.nii.gz', '_p{}.nii.gz'.format(group_index))
                                writter = sitk.ImageFileWriter()
                                writter.SetFileName(save_name)
                                writter.Execute(sitk.GetImageFromArray(grayscale_cam[i][group_index]))
                                # [batch, organ_groups, z, y, x, channel] to [batch, organ_groups, z, y, x]
                                # TODO currently we just use the second layer of input:
                                writter.SetFileName(origin_save_name)
                                writter.Execute(sitk.GetImageFromArray(origin_img[i][group_index][1]))
                    else:
                        raise ValueError(f'not supported shape: {grayscale_cam[i].shape}') 
                # ---------------------------------------  cam create  --------------------------------------- #

                # --------------------------------------  cam evaluate  -------------------------------------- #
                if eval_func in ['corr', 'corr_logit']:
                    grayscale_cam = np.array(grayscale_cam)  # # grayscale_cam -- 16 * [1, 256, 256] - batch * [1, 256, 256]
                    if tanh_flag and data_max_value:
                        para_k = (np.arctanh(t_max) - np.arctanh(t_min))/(data_max_value-data_min_value)
                        para_b = (np.arctanh(t_max)*data_min_value-np.arctanh(t_min)*data_max_value)/(data_min_value-data_max_value)
                        grayscale_cam = (np.arctanh(grayscale_cam+1e-10) - para_b)/(para_k + 1e-10)
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
                    channel_per_group = x.shape[1] // self.groups
                    for gc in range(self.groups):
                        extended_cam[:, gc*channel_per_group:(gc+1)*channel_per_group, :] = np.expand_dims(grayscale_cam[:, gc, :], axis=1)
                    # extended_cam: numpy [batch, in_channel, length, width]
                    cam_input = torch.from_numpy(extended_cam).to(device) * x
                    cam_input = cam_input_normalization(cam_input)
                    cam_pred = model(cam_input)
                    if eval_func == 'basic':
                        origin_category, single_origin_confidence = predict_category, pred_score
                        _, single_cam_confidence, _ = pred_score_calculator(x.shape[0], cam_pred, creator_tc,
                                                                                    origin_pred_category=origin_category,
                                                                                    out_logit=False)
                        single_drop = torch.relu(torch.from_numpy(single_origin_confidence\
                                    - single_cam_confidence)).div(torch.from_numpy(single_origin_confidence) + 1e-7)
                    elif eval_func == 'logit':
                        origin_category, single_origin_confidence = y, pred_score
                        _, single_cam_confidence, single_cam_nega_scores = pred_score_calculator(x.shape[0], cam_pred, 'GT',
                                                                                            origin_pred_category=origin_category)
                        
                        single_drop = nega_score > single_cam_nega_scores  # 新的drop越大越好
                    acc_ori.extend(np.argmax(softmax(cam_pred.cpu().data.numpy(), axis=-1),axis=-1))
                    acc_cam.extend(predict_category)
                    acc_gt.extend(y.item())
                    counter += x.shape[0]
                    single_increase = single_origin_confidence < single_cam_confidence
                    increase += single_increase.sum().item()
                    drop += single_drop.sum().item()
                # --------------------------------------  cam evaluate  -------------------------------------- #

                # early stop
                if self.max_iter is not None:
                    if counter >= self.max_iter:
                        print('counter meet max iter')
                        break

        # --------------------------------------  cam evaluate  -------------------------------------- #
        if eval_func in ['corr', 'corr_logit']:
            print('total samples:', counter)
            # cam分数和类别的AUROC，代表的是cam正确反映分类情况的能力
            # for mutliclasses, use pos-neg to calculate
            if self.num_classes>2:
                reg_corr_cate_matrix = []
                for item in corr_cate_matrix:
                    if item==creator_tc:
                        reg_corr_cate_matrix.append(1)
                    else:
                        reg_corr_cate_matrix.append(0)
                corr_cate_matrix = reg_corr_cate_matrix
            corr_cam_matrix = np.nan_to_num(corr_cam_matrix, copy=False, nan=0, posinf=0, neginf=0)
            auc = roc_auc_score(corr_cate_matrix, corr_cam_matrix)
            print('outlier rate-- AUROC of <CAM & Label>: ', auc)
            corr_dir = cam_dir.replace('./output/cam/', './output/figs/')
            if not os.path.exists(corr_dir):
                os.makedirs(corr_dir)
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
            corr_dir = cam_dir.replace('./output/cam/', './output/figs/')
            if not os.path.exists(corr_dir):
                os.makedirs(corr_dir)
            eval_borl_save_name = os.path.join(corr_dir, f'eval_with_{eval_func}.txt')
            text_save(eval_borl_save_name, avg_increase, avg_drop, counter)
        # --------------------------------------  cam evaluate  -------------------------------------- #
