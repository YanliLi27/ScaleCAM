import os
from torch.utils.data import DataLoader
from typing import Union
import numpy as np
# analyzer
from cam_components.agent.analyzer_funcs import cam_stats_step
from cam_components.agent.im_funcs import im_save
# predictor
from cam_components.agent.creator_funcs import cam_creator_step
from cam_components.agent.utils.target_cam_calculation import target_cam_selection
from cam_components.agent.im_funcs import im_reader, maxmin_reader
# cam methods
from cam_components.methods import GradCAM_A, GradCAM_P, FullCAM_A, FullCAM_P, GradCAMPP_A, GradCAMPP_P, XGradCAM_A, XGradCAM_P


class CAMAgent:
    def __init__(self,
                # ---------------- model and dataset -------------#
                model, target_layer, dataset:Union[DataLoader, np.array],  
                num_out_channel:int=512, num_classes:int=2,  
                groups:int=1, fold_order:int=0,  
                # ---------------- model and dataset -------------#
                cam_method:str='gradcam', im_dir:str='./output/im/taskname', cam_dir:str='./output/cam/taskname',
                # cam method and im paths and cam output
                batch_size:int=1, target_category=1,  # info of the running process
                maxmin_flag:bool=False, remove_minus_flag:bool=True, # creator
                im_selection_mode:str='all', im_selection_extra:float=0.05, # importance matrices attributes
                max_iter=None,  # early stop
                randomization:bool=False, random_severity:int=1,  # model randomization for sanity check
                ) -> None:
        

        # cam info
        cam_method_zoo = {"gradcam": [GradCAM_A, GradCAM_P], 
                        "fullcam": [FullCAM_A, FullCAM_P],
                        "gradcampp":[GradCAMPP_A, GradCAMPP_P],
                        "xgradcam":[XGradCAM_A, XGradCAM_P]}
        self.cam_method = cam_method_zoo[cam_method]

        # model info
        self.model = model
        self.target_layer = target_layer
        self.num_out_channel = num_out_channel
        self.batch_size = batch_size

        # dataset info
        self.num_classes = num_classes
        self.dataset = dataset

        assert (target_category in ['GT', None] or type(target_category) == int)
        self.target_category = target_category  # targeted category
        self.groups = groups  # group convolution
        self.fold_order = fold_order  # if use cross-validation

        # Extra function of the CAM creator
        self.maxmin_flag = maxmin_flag
        self.remove_minus_flag = remove_minus_flag
        assert im_selection_mode in ['max', 'top', 'diff_top', 'freq', 'index', 'all']
        self.im_selection_mode = im_selection_mode
        self.im_selection_extra = im_selection_extra
        
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
        im_name = f'All_fold{self.fold_order}_im_cate{self.target_category}_{cam_method}.csv' 
        im_path = os.path.join(im_dir, im_name)
        self.im_path = im_path

        cam_son_dir = f'fold{self.fold_order}_cate{self.target_category}_mt{cam_method}'
        cam_sub_dir = f'norm{self.maxmin_flag}_rm0{self.remove_minus_flag}_{self.im_selection_mode}{self.im_selection_extra}'
        self.cam_dir = os.path.join(cam_dir, cam_son_dir, cam_sub_dir)
        if not os.path.exists(self.cam_dir):
            os.makedirs(self.cam_dir)
        print('im_path: {}'.format(im_path))
        print('cam output dir: {}'.format(self.cam_dir))
        print(f'------------------------------ initialized ------------------------------')


    def analyzer_main(self):
        if not os.path.isfile(self.im_path):  # only when the file dosen't exist -- because some loop would be repeated in experiments
            print('--------- creating IMs ---------')
            im_overall, im_target, im_diff, cam_grad_max_matrix, cam_grad_min_matrix \
                = cam_stats_step(self.cam_method[0], self.target_layer, # for the cam setting
                                self.model, self.dataset, self.num_out_channel, self.num_classes, # for the model and dataset
                                target_category=self.target_category,
                                batch_size=self.batch_size
                                )
            im_save(im_overall, im_target, im_diff, cam_grad_max_matrix, cam_grad_min_matrix,
                    self.im_path)
        


    def creator_main(self, eval_act:str='false', mm_ratio:float=1.5, use_origin:bool=True):
        '''
        mm_ratio for better visuaization
        use_origin for overlay/or not
        '''
            
        # step 1. im_read - or not
        print('--------- creating CAMs ---------')
        if os.path.exists(self.im_path) and self.im_selection_mode != 'all':
            print('loading importance matrix with mode:{}'.format(self.im_selection_mode))
            im = im_reader(self.im_path, self.im_selection_mode)
            # im [num_classes, num_features]
            im = target_cam_selection(im, mode=self.im_selection_mode, extra=self.im_selection_extra)
            # im [num_classes, num_features]
        else:
            im = None
        
        # step 2. max-min normalization - or not
        if self.im_path and self.maxmin_flag:
            data_max_value, data_min_value = maxmin_reader(self.im_path, self.target_category)
            data_max_value, data_min_value = data_max_value * mm_ratio, data_min_value * mm_ratio
        else:
            data_max_value, data_min_value = None, None
        
        # step 3. pred step
        cam_creator_step(self.cam_method[1], self.model, self.target_layer, self.dataset, self.cam_dir,  # required attributes
                        # optional function:
                        im=im, data_max_value=data_max_value, data_min_value=data_min_value, remove_minus_flag=self.remove_minus_flag,
                        max_iter=self.max_iter, use_origin=use_origin,
                        batch_size=self.batch_size, groups=self.groups, target_category=self.target_category,
                        fold_order=self.fold_order,
                        eval_func=eval_act
                        )


