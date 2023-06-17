from typing import Union
from cam_components.agent_main import CAMAgent
from torch.utils.data import DataLoader
import os


def naturalimage_runner(target_category=None, model_flag:str='resnet', task:str='CatsDogs', dataset_split:str='val',
                        max_iter=None, randomization=False, random_severity=0):
    # -------------------------------- optional: -------------------------------- #
    batch_size:int=16
    target_category:Union[None, int, str]=1  # info of the running process
    # more functions
    im_selection_extra:float=0.05  # importance matrices attributes
    max_iter=max_iter  # early stop
    groups:int=1
    # -------------------------------- optional end -------------------------------- #


    # information needed:
    from predefined.natural_components.main_generator import main_generator
    model, target_layer, dataset, im_dir, cam_dir, num_out_channel, num_classes = \
                                                    main_generator(model_flag=model_flag,
                                                                    task=task,
                                                                    dataset_split=dataset_split,
                                                                    fold_order=0, 
                                                                    randomization=randomization, random_severity=random_severity
                                                                    )
    dataset = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False,
                            num_workers=4, pin_memory=True)
    
    use_origin = False if task=='MNIST' else True
    fold_order:int=0

    # -------------------------------- start loop -------------------------------- #

    cam_method_zoo = ['gradcam', 'fullcam', 'gradcampp', 'xgradcam']
    maxmin_flag_zoo = [True, False]  # intensity scaling
    remove_minus_flag_zoo = [True, False]  # remove the part below zero, default: True in the original Grad CAM
    im_selection_mode_zoo = ['all', 'diff_top']  # use feature selection or not -- relied on the importance matrices

    for method in cam_method_zoo:
        for im in im_selection_mode_zoo:
            for mm in maxmin_flag_zoo:
                for rm in remove_minus_flag_zoo:
                    Agent = CAMAgent(model, target_layer, dataset,  
                                    num_out_channel, num_classes,  
                                    groups, fold_order,  
                                    # optional:
                                    cam_method=method, im_dir=im_dir, cam_dir=cam_dir, # cam method and im paths and cam output
                                    batch_size=batch_size, target_category=target_category,  # info of the running process
                                    maxmin_flag=mm, remove_minus_flag=rm, # creator
                                    im_selection_mode=im, im_selection_extra=im_selection_extra, # importance matrices attributes
                                    max_iter=max_iter,  # early stop
                                    randomization=False, random_severity=0  # model randomization for sanity check
                                    )
                    Agent.analyzer_main()
                    Agent.creator_main(eval_act='corr', mm_ratio=1.5, use_origin=use_origin)


def catsdog3d_runner(target_category=None, model_flag:str='resnet', task:str='catsdogs3d', dataset_split:str='val'):    
    # -------------------------------- optional: -------------------------------- #
    batch_size:int=16
    target_category:Union[None, int, str]=1  # info of the running process
    # more functions
    im_selection_extra:float=0.05  # importance matrices attributes
    max_iter=max_iter  # early stop
    groups:int=1
    # -------------------------------- optional end -------------------------------- #


    # information needed:
    from predefined.multi_components.generators.catsdogs3d_generator import get_data_weight_output_path
    model, train_dataset, val_dataset, im_dir, cam_dir, target_layer, num_out_channel, num_classes =\
              get_data_weight_output_path(task_name=task)
    if dataset_split == 'val':
        dataset = val_dataset
    else:
        dataset = train_dataset
    dataset = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False,
                    num_workers=4, pin_memory=True)
    

    use_origin = False if task=='MNIST' else True
    fold_order:int=0

    # -------------------------------- start loop -------------------------------- #

    cam_method_zoo = ['gradcam', 'fullcam', 'gradcampp', 'xgradcam']
    maxmin_flag_zoo = [True, False]  # intensity scaling
    remove_minus_flag_zoo = [True, False]  # remove the part below zero, default: True in the original Grad CAM
    im_selection_mode_zoo = ['all', 'diff_top']  # use feature selection or not -- relied on the importance matrices

    for method in cam_method_zoo:
        for im in im_selection_mode_zoo:
            for mm in maxmin_flag_zoo:
                for rm in remove_minus_flag_zoo:
                    Agent = CAMAgent(model, target_layer, dataset,  
                                    num_out_channel, num_classes,  
                                    groups, fold_order,  
                                    # optional:
                                    cam_method=method, im_dir=im_dir, cam_dir=cam_dir, # cam method and im paths and cam output
                                    batch_size=batch_size, target_category=target_category,  # info of the running process
                                    maxmin_flag=mm, remove_minus_flag=rm, # creator
                                    im_selection_mode=im, im_selection_extra=im_selection_extra, # importance matrices attributes
                                    max_iter=max_iter,  # early stop
                                    randomization=False, random_severity=0  # model randomization for sanity check
                                    )
                    Agent.analyzer_main()
                    Agent.creator_main(eval_act='corr', mm_ratio=1.5, use_origin=use_origin)
    

def medical_runner(target_category=None, model_flag:str='resnet', task:str='ddsm', dataset_split:str='val'):
    # -------------------------------- optional: -------------------------------- #
    batch_size:int=16
    target_category:Union[None, int, str]=1  # info of the running process
    # more functions
    im_selection_extra:float=0.05  # importance matrices attributes
    max_iter=None  # early stop
    groups:int=1  # no group convolution here
    # -------------------------------- optional end -------------------------------- #

    # information needed:
    from predefined.multi_components.generators.multi_task_generators import get_data_weight_output_path
    for i in range(0, 1):
        fold_order = i
        model, train_dataset, val_dataset, im_dir, cam_dir, target_layer, num_out_channel, num_classes =\
              get_data_weight_output_path(task_name=task)
        if dataset_split == 'val':
            dataset = val_dataset
        else:
            dataset = train_dataset
        dataset = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False,
                        num_workers=4, pin_memory=True)

        

        # -------------------------------- start loop -------------------------------- #

        cam_method_zoo = ['gradcam', 'fullcam', 'gradcampp', 'xgradcam']
        maxmin_flag_zoo = [True, False]  # intensity scaling
        remove_minus_flag_zoo = [True, False]  # remove the part below zero, default: True in the original Grad CAM
        im_selection_mode_zoo = ['all', 'diff_top']  # use feature selection or not -- relied on the importance matrices

        for method in cam_method_zoo:
            for im in im_selection_mode_zoo:
                for mm in maxmin_flag_zoo:
                    for rm in remove_minus_flag_zoo:
                        Agent = CAMAgent(model, target_layer, dataset,  
                                        num_out_channel, num_classes,  
                                        groups, fold_order,  
                                        # optional:
                                        cam_method=method, im_dir=im_dir, cam_dir=cam_dir, # cam method and im paths and cam output
                                        batch_size=batch_size, target_category=target_category,  # info of the running process
                                        maxmin_flag=mm, remove_minus_flag=rm, # creator
                                        im_selection_mode=im, im_selection_extra=im_selection_extra, # importance matrices attributes
                                        max_iter=max_iter,  # early stop
                                        randomization=False, random_severity=0  # model randomization for sanity check
                                        )
                        Agent.analyzer_main()
                        Agent.creator_main(eval_act='corr', mm_ratio=1.5, use_origin=True)


def esmira_runner(method:str='gradcam', target_category=None, data_dir:str='D:\\ESMIRA\\ESMIRA_common',
                target_catename:list=['EAC','ATL'], target_site:list=['Wrist'], target_dirc:list=['TRA', 'COR']):
    # -------------------------------- optional: -------------------------------- #
    batch_size:int=16
    target_category:Union[None, int, str]=1  # info of the running process
    # more functions
    im_selection_extra:float=0.05  # importance matrices attributes
    max_iter=None  # early stop
    groups:int=len(target_dirc) * len(target_site)
    # -------------------------------- optional end -------------------------------- #

    # information needed:
    from predefined.esmira_components.generators.dataset_class import ESMIRA_generator
    from predefined.esmira_components.model import ModelClass
    from predefined.esmira_components.weight_path import output_finder
    import torch

    dataset_generator = ESMIRA_generator(data_dir, target_catename, target_site, target_dirc)
    for fold_order in range(5):
        _, target_dataset = dataset_generator.returner(phase='train', fold_order=fold_order, mean_std=False)
        dataset = DataLoader(dataset=target_dataset, batch_size=batch_size, shuffle=False,
            num_workers=4, pin_memory=True)
        # input: [N*5, 512, 512] + int(label)

        # Step. 2 get the model: (can be any nn.Module, make sure it fit your input size and output size)
        
        # --------------------------------------- model --------------------------------------- #
        in_channel = groups * 5
        model = ModelClass(in_channel, num_classes=2)
        weight_path = output_finder(target_catename, target_site, target_dirc, fold_order)
        weight_abs_path = os.path.join('D:\\ESMIRAcode\\RA_Class\\models\\weights', weight_path)
        if os.path.isfile(weight_abs_path):
            checkpoint = torch.load(weight_abs_path)
            model.load_state_dict(checkpoint)
        else:
            raise ValueError('weights not exisst')
        target_layer = [model.encoder_class.Conv4]
        # --------------------------------------- model --------------------------------------- #

        # --------------------------------------- im --------------------------------------- #
        num_out_channel = 256 * groups
        num_classes = 2
        im_dir = './output/im/ESMIRA_{}'.format(weight_path.replace('.model', ''))
        if not os.path.exists(im_dir):
            os.makedirs(im_dir)
        # --------------------------------------- im --------------------------------------- #
        cam_dir = './output/cam/ESMIRA_{}'.format(weight_path.replace('.model', ''))


        # -------------------------------- start loop -------------------------------- #
        cam_method_zoo = ['gradcam', 'fullcam', 'gradcampp', 'xgradcam']
        maxmin_flag_zoo = [True, False]  # intensity scaling
        remove_minus_flag_zoo = [True, False]  # remove the part below zero, default: True in the original Grad CAM
        im_selection_mode_zoo = ['all', 'diff_top']  # use feature selection or not -- relied on the importance matrices

        for method in cam_method_zoo:
            for im in im_selection_mode_zoo:
                for mm in maxmin_flag_zoo:
                    for rm in remove_minus_flag_zoo:
                        Agent = CAMAgent(model, target_layer, dataset,  
                                        num_out_channel, num_classes,  
                                        groups, fold_order,  
                                        # optional:
                                        cam_method=method, im_dir=im_dir, cam_dir=cam_dir, # cam method and im paths and cam output
                                        batch_size=batch_size, target_category=target_category,  # info of the running process
                                        maxmin_flag=mm, remove_minus_flag=rm, # creator
                                        im_selection_mode=im, im_selection_extra=im_selection_extra, # importance matrices attributes
                                        max_iter=max_iter,  # early stop
                                        randomization=False, random_severity=0  # model randomization for sanity check
                                        )
                        Agent.analyzer_main()
                        Agent.creator_main(eval_act='corr', mm_ratio=1.5, use_origin=True)