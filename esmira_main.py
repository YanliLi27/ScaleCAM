import os
from main_func.analyzer import analyzer_main
from main_func.predictor import predictor_main
from main_func.evaluator import evaluator_main
from cam_components import GradCAM_A, GradCAM_P, FullCAM_A, FullCAM_P, GradCAMPP_A, GradCAMPP_P, XGradCAM_A, XGradCAM_P
import argparse
from esmira_components.generators.dataset_class import ESMIRA_generator
from esmira_components.model import ModelClass
from esmira_components.weight_path import output_finder
import torch


def get_args():
    parser = argparse.ArgumentParser()
    # general args
    parser.add_argument('--batch_size', type=int, default=16, help='Dataloader batch size')
    parser.add_argument('--groups', type=int, default=1, help='If use group convolution')
    parser.add_argument('--task', type=str, default='CatsDogs', choices=['CatsDogs', 'MNIST', 'Imagenet'],
                         help='tasks')  # for model, dataset, cam, and file management
    
    # for model
    parser.add_argument('--model_flag', type=str, default='resnet', choices=['resnet', 'vgg', 'scratch', 'scratch_mn'],
                         help='model type')
    parser.add_argument('--randomization', type=bool, default=False, help='If start randomization')
    parser.add_argument('--random_severity', type=int, default=0, choices=[0, 1, 2, 3, 4], help='n/4 randomization')

    # for file management and loop
    parser.add_argument('--fold_order', type=int, default=0, help='For cross validation')
    parser.add_argument('--dataset_split', type=str, default='val', choices=['train', 'val', 'test'],
                         help='split')


    # cam required args
    parser.add_argument('--target_category_flag', default=None, help='Output category')
    parser.add_argument('--method', type=str, default='gradcam', choices=['gradcam', 'fullcam', 'gradcampp', 'smoothgradcam'],
                        help='If use the mean of gradients')
    parser.add_argument('--confidence_weight_flag', type=bool, default=False, help='Output category')

    # optional function -- core improvements
    parser.add_argument('--use_stat', type=bool, default=True, help='For using analyzer and im')
    # for predictor
    parser.add_argument('--stat_maxmin_flag', type=bool, default=True, help='For maxmin normalization')
    parser.add_argument('--remove_minus_flag', type=bool, default=True, help='If remove the part of heatmaps less than 0')
    parser.add_argument('--im_selection_mode', type=str, default='diff_top', choices=['max', 'top', 'diff_top', 'freq', 'index', 'all'],
                        help='If use statistic')
    parser.add_argument('--im_selection_extra', type=float, default=0.05, help='attributes of selection mode')
    parser.add_argument('--max_iter', default=None, help='max iteration to save time')

    args = parser.parse_args()
    return args


def esmira_analyzer_process(method:str='gradcam', target_category=None, data_dir:str='D:\\ESMIRA\\ESMIRA_common',
                            target_catename:list=['EAC','ATL'], target_site:list=['Wrist'], target_dirc:list=['TRA', 'COR']):
    general_args = get_args()
    # --------------------------------------- re-define for loop --------------------------------------- #
    # Select the CAM method and target_category (optional)
    general_args.method = method
    general_args.target_category_flag = target_category
    # The model, task(data) and dataset split
    general_args.dataset_split = 'val'
    general_args.groups = len(target_dirc) * len(target_site)
    general_args.batch_size = 1

    # --------------------------------------- re-define for loop --------------------------------------- #

    dataset_generator = ESMIRA_generator(data_dir, target_catename, target_site, target_dirc)
    for fold_order in range(5):
        general_args.fold_order = fold_order
        _, target_dataset = dataset_generator.returner(phase='train', fold_order=fold_order, mean_std=False)
        # input: [N*5, 512, 512] + int(label)

        # Step. 2 get the model: (can be any nn.Module, make sure it fit your input size and output size)
        
        # --------------------------------------- model --------------------------------------- #
        in_channel = general_args.groups * 5
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
        out_channel = 256 * general_args.groups
        num_classes = 2
        im_dir = os.path.join('./output/im/ESMIRA')
        if not os.path.exists(im_dir):
            os.makedirs(im_dir)
        im_name = '{}_{}'.format(general_args.method, weight_path.replace('.model', '.csv')) 
        im_path = os.path.join(im_dir, im_name)
        # --------------------------------------- im --------------------------------------- #
            

        # im_path: # ./output/im/MNIST_val_resnet/All_fold0_im_cateNone_Grad.csv

        cam_method_zoo = {"gradcam": [GradCAM_A, GradCAM_P], 
        "fullcam": [FullCAM_A, FullCAM_P],
        "gradcampp":[GradCAMPP_A, GradCAMPP_P],
        "xgradcam":[XGradCAM_A, XGradCAM_P]}
        cam_method = cam_method_zoo[general_args.method]
        analyzer_main(cam_method[0], model, target_layer, target_dataset, im_path, out_channel, num_classes, general_args) 



def esmira_predictor_process(method:str='gradcam', target_category=None, data_dir:str='D:\\ESMIRA\\ESMIRA_common',
                            target_catename:list=['EAC','ATL'], target_site:list=['Wrist'], target_dirc:list=['TRA', 'COR'],
                            randomization:bool=False, random_severity:int=0,
                            maxmin_flag:bool=True, remove_minus_flag:bool=True, im_selection_mode:str='diff_top', im_selection_extra:float=0.05,
                            max_iter=None,
                            ):
    general_args = get_args()
    # --------------------------------------- re-define for loop --------------------------------------- #
    general_args.batch_size = 1
    # Select the CAM method and target_category (optional)
    general_args.method = method
    general_args.target_category_flag = target_category
    # Extra function of the CAM creator
    general_args.stat_maxmin_flag = maxmin_flag
    general_args.remove_minus_flag = remove_minus_flag
    general_args.im_selection_mode = im_selection_mode
    general_args.im_selection_extra = im_selection_extra
    general_args.max_iter = max_iter
    # 是否模型随机化
    general_args.randomization = randomization
    general_args.random_severity = random_severity

    # The model, task(data) and dataset split
    general_args.dataset_split = 'val'
    general_args.groups = len(target_dirc) * len(target_site)
    # --------------------------------------- re-define for loop --------------------------------------- #

    dataset_generator = ESMIRA_generator(data_dir, target_catename, target_site, target_dirc)
    for fold_order in range(5):
        general_args.fold_order = fold_order
        _, target_dataset = dataset_generator.returner(phase='train', fold_order=fold_order, mean_std=False)        
        # --------------------------------------- model --------------------------------------- #
        in_channel = general_args.groups * 5
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
        im_dir = os.path.join('./output/im/ESMIRA')
        if not os.path.exists(im_dir):
            os.makedirs(im_dir)
        im_name = '{}_{}'.format(general_args.method, weight_path.replace('.model', '.csv')) 
        im_path = os.path.join(im_dir, im_name)
        # --------------------------------------- im --------------------------------------- #
        # --------------------------------------- cam dir--------------------------------------- #
        cam_father_dir = os.path.join('./output/cam/', weight_path.replace('.model', '_dir'))
        cam_son_dir = f'fold{general_args.fold_order}_cate{general_args.target_category_flag}_mt{general_args.method}'
        cam_sub_dir = f'norm{general_args.stat_maxmin_flag}_rm0{general_args.remove_minus_flag}_{general_args.im_selection_mode}{general_args.im_selection_extra}'
        cam_dir = os.path.join(cam_father_dir, cam_son_dir, cam_sub_dir)
        if not os.path.exists(cam_dir):
            os.makedirs(cam_dir)
        # --------------------------------------- cam dir--------------------------------------- #


        cam_method_zoo = {"gradcam": [GradCAM_A, GradCAM_P], 
                        "fullcam": [FullCAM_A, FullCAM_P],
                        "gradcampp":[GradCAMPP_A, GradCAMPP_P],
                        "xgradcam":[XGradCAM_A, XGradCAM_P]}
        cam_method = cam_method_zoo[general_args.method]
        predictor_main(cam_method[1], model, target_layer, target_dataset, cam_dir, general_args, im_path, mm_ratio=1)


def esmira_evaluator_process(method:str='gradcam', target_category=None, data_dir:str='D:\\ESMIRA\\ESMIRA_common',
                            target_catename:list=['EAC','ATL'], target_site:list=['Wrist'], target_dirc:list=['TRA', 'COR'],
                            randomization:bool=False, random_severity:int=0,
                            maxmin_flag:bool=True, remove_minus_flag:bool=True,
                            im_selection_mode:str='diff_top', im_selection_extra:float=0.05,
                            max_iter=None, fold:int=5
                        ):
    general_args = get_args()
    # --------------------------------------- re-define for loop --------------------------------------- #
    general_args.batch_size = 1
    # Select the CAM method and target_category (optional)
    general_args.method = method
    general_args.target_category_flag = target_category

    # Extra function of the CAM creator
    general_args.stat_maxmin_flag = maxmin_flag
    general_args.remove_minus_flag = remove_minus_flag
    general_args.im_selection_mode = im_selection_mode
    general_args.im_selection_extra = im_selection_extra
    if max_iter is not None:
        general_args.max_iter = max_iter
    # 是否模型随机化
    general_args.randomization = randomization
    general_args.random_severity = random_severity
    # The model, task(data) and dataset split
    general_args.dataset_split = 'val'
    general_args.groups = len(target_dirc) * len(target_site)
    # --------------------------------------- re-define for loop --------------------------------------- #

    dataset_generator = ESMIRA_generator(data_dir, target_catename, target_site, target_dirc)
    for fold_order in range(fold):
        general_args.fold_order = fold_order
        _, target_dataset = dataset_generator.returner(phase='train', fold_order=fold_order, mean_std=False)        
        # --------------------------------------- model --------------------------------------- #
        in_channel = general_args.groups * 5
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
        im_dir = os.path.join('./output/im/ESMIRA')
        if not os.path.exists(im_dir):
            os.makedirs(im_dir)
        im_name = '{}_{}'.format(general_args.method, weight_path.replace('.model', '.csv')) 
        im_path = os.path.join(im_dir, im_name)
        # --------------------------------------- im --------------------------------------- #
        # --------------------------------------- cam dir--------------------------------------- #
        cam_father_dir = os.path.join('./output/cam/', weight_path.replace('.model', '_dir'), '{}_{}_{}')
        cam_son_dir = f'fold{general_args.fold_order}_cate{general_args.target_category_flag}_mt{general_args.method}'
        cam_sub_dir = f'norm{general_args.stat_maxmin_flag}_rm0{general_args.remove_minus_flag}_{general_args.im_selection_mode}{general_args.im_selection_extra}'
        cam_dir = os.path.join(cam_father_dir, cam_son_dir, cam_sub_dir)
        if not os.path.exists(cam_dir):
            os.makedirs(cam_dir)


        print('im_path: {}'.format(im_path))
        cam_method_zoo = {"gradcam": [GradCAM_A, GradCAM_P], 
        "fullcam": [FullCAM_A, FullCAM_P],
        "gradcampp":[GradCAMPP_A, GradCAMPP_P],
        "xgradcam":[XGradCAM_A, XGradCAM_P]}
        cam_method = cam_method_zoo[general_args.method]
        evaluator_main(cam_method[1], model, target_layer, target_dataset, general_args, im_path, version='corr')


if __name__ == '__main__':
    methods = ['fullcam'] 
    # for method in methods:
    #     esmira_analyzer_process(method=method, target_category=1, data_dir='D:\\ESMIRA\\ESMIRA_common',
    #                             target_catename=['CSA'], target_site=['Wrist'], target_dirc=['TRA', 'COR'])

    maxmin = [True, False]
    rmove = [True, False]
    # maxmin = [False]
    # rmove = [True]
    immode = ['all', 'diff_top']
    
    for method in methods:
        for mm in maxmin:
            for rm in rmove:
                for im in immode:
                    # esmira_predictor_process(method=method, target_category=1, data_dir='D:\\ESMIRA\\ESMIRA_common',
                    #         target_catename=['EAC'], target_site=['Wrist'], target_dirc=['TRA', 'COR'],
                    #         randomization=False, random_severity=0,
                    #         maxmin_flag=mm, remove_minus_flag=rm, im_selection_mode=im, im_selection_extra=0.3,
                    #         max_iter=None)

                    esmira_evaluator_process(method=method, target_category=1, data_dir='D:\\ESMIRA\\ESMIRA_common',
                            target_catename=['EAC', 'ATL'], target_site=['Wrist'], target_dirc=['TRA', 'COR'],
                            randomization=False, random_severity=0,
                            maxmin_flag=mm, remove_minus_flag=rm,
                            im_selection_mode=im, im_selection_extra=0.3,
                            max_iter=None, fold=5
                                        )

    # esmira_predictor_process(method='xgradcam', target_category=None, model_flag='vgg', task='Imagenet', dataset_split='val',
    #                     randomization=False, random_severity=0,
    #                     maxmin_flag=False, remove_minus_flag=True, im_selection_mode='diff_top', im_selection_extra=0.05,
    #                     max_iter=4800,
    #                     )


    # esmira_evaluator_process(method='gradcam', target_category=None, model_flag='vgg', task='Imagenet', dataset_split='val',
    #                     randomization=False, random_severity=0,
    #                     maxmin_flag=False, remove_minus_flag=True, im_selection_mode='all', im_selection_extra=0.05,
    #                     max_iter=None,
    #                     )