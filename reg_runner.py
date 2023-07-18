from typing import Union
from cam_components.agent_main import CAMAgent
from torch.utils.data import DataLoader
import os
from torchsummary import summary


def ramris_pred_runner(data_dir='', target_category:Union[None, int, str, list]=None, 
                 target_site=['Wrist'], target_dirc=['TRA', 'COR'],
                 target_biomarker=['SYN'],
                 target_reader=['Reader1', 'Reader2'], task_mode='clip', phase='train',
                 full_img:bool=True,
                 target_output:Union[None, int, str, list]=[0]):
    # -------------------------------- optional: -------------------------------- #
    batch_size:int=2
    target_category:Union[None, int, str, list]=target_category  # info of the running process
    # more functions
    im_selection_extra:float=0.05  # importance matrices attributes
    max_iter=None  # early stop
    groups:int=len(target_dirc) * len(target_site)
    ram:bool=True  # if it's a regression task
    use_pred:bool=False
    # -------------------------------- optional end -------------------------------- #

    # information needed:
    from predefined.ramris_components.models.clip_model import ModelClip
    from predefined.ramris_components.generators.utli_generator import ESMIRA_generator
    from predefined.ramris_components.utils.output_finder import output_finder
    import torch

    if target_biomarker:
        for item in target_biomarker:
            assert (item in ['ERO', 'BME', 'SYN', 'TSY'])
    dataset_generator = ESMIRA_generator(data_dir, target_category, target_site, target_dirc, target_reader, target_biomarker, task_mode)

    for fold_order in range(0, 1):
        _, val_dataset = dataset_generator.returner(task_mode=task_mode, phase=phase, fold_order=fold_order,
                                                                material='img', monai=True, full_img=full_img)
        dataset = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False,
            num_workers=4, pin_memory=True)
        # input: [N*5, 512, 512] + int(label)

        # Step. 2 get the model: (can be any nn.Module, make sure it fit your input size and output size)
        if full_img:
            depth = 20
        else:
            depth = 5
        in_channel = len(target_site) * len(target_dirc) * depth   # input is a (5*site*dirc) * 512 * 512 img
        out_ch = 0
        output_matrix = [[15, 15, 3, 10],[8, 8, 4, 8],[10, 10, 5, 10]]
        site_order = {'Wrist':0, 'MCP':1, 'Foot':2}
        bio_order = {'ERO':0, 'BME':1, 'SYN':2, 'TSY':3}
        for site in target_site:
            if target_biomarker is None:
                target_biomarker = ['ERO', 'BME', 'SYN', 'TSY']
            for biomarker in target_biomarker:
                out_ch += output_matrix[site_order[site]][bio_order[biomarker]]
        width = 2
        model = ModelClip(in_channel, out_ch=out_ch, dimension=2, group_cap=depth, width=2)  
        summary(model, (40, 512, 512))

        weight_path = output_finder(target_category, target_site, target_dirc, fold_order)
        mid_path = 'ALLBIO' if (target_category is None or len(target_category)>1) else f'ALL{target_category[0]}'
        weight_abs_path = os.path.join(f'D:\\ESMIRAcode\\RA_CLIP\\models\\weights\\{mid_path}', weight_path)
        if os.path.isfile(weight_abs_path):
            checkpoint = torch.load(weight_abs_path)
            model.load_state_dict(checkpoint)
        else:
            raise ValueError('weights not exist')
 
        target_layer = [model.encoder_class.Conv4]
        # --------------------------------------- model --------------------------------------- #

        # --------------------------------------- im --------------------------------------- #
        num_out_channel = 256 * groups * width
        num_classes = out_ch
        im_dir = './output/im/RAMRISpred_{}'.format(weight_path.replace('.model', ''))
        if not os.path.exists(im_dir):
            os.makedirs(im_dir)
        # --------------------------------------- im --------------------------------------- #
        cam_dir = './output/cam/RAMRISpred_{}'.format(weight_path.replace('.model', ''))

        # -------------------------------- start loop -------------------------------- #
        cam_method_zoo = ['gradcam']#, 'fullcam', 'gradcampp', 'xgradcam']
        maxmin_flag_zoo = [True, False]  # intensity scaling
        remove_minus_flag_zoo = [False, True]  # remove the part below zero, default: True in the original Grad CAM
        im_selection_mode_zoo = ['all']#, 'diff_top']  # use feature selection or not -- relied on the importance matrices

        for method in cam_method_zoo:
            for im in im_selection_mode_zoo:
                for mm in maxmin_flag_zoo:
                    for rm in remove_minus_flag_zoo:
                        if mm:
                            tanh_flag = True
                        else:
                            tanh_flag = False
                        Agent = CAMAgent(model, target_layer, dataset,  
                                        num_out_channel, num_classes,  
                                        groups, fold_order, ram,
                                        # optional:
                                        cam_method=method, im_dir=im_dir, cam_dir=cam_dir, # cam method and im paths and cam output
                                        batch_size=batch_size, target_category=target_output,  # info of the running process
                                        maxmin_flag=mm, remove_minus_flag=rm, # creator
                                        im_selection_mode=im, im_selection_extra=im_selection_extra, # importance matrices attributes
                                        max_iter=max_iter,  # early stop
                                        randomization=False, random_severity=0,  # model randomization for sanity check
                                        use_pred=use_pred
                                        )
                        Agent.analyzer_main()
                        Agent.creator_main(eval_act=False, mm_ratio=2, use_origin=True, 
                                           cluster=[15, 3, 10], cluster_start=15,
                                           tanh_flag=tanh_flag)