from predefiend_runner import naturalimage_runner, catsdog3d_runner, esmira_runner, medical_runner
from reg_runner import ramris_pred_runner


if __name__ == '__main__':
    # for test 
    task_zoo = ['Imagenet'] #['MNIST', 'Imagenet','CatsDogs'] 
    model_zoo = {'CatsDogs':'vgg', 'Imagenet':'resnet', 'MNIST':'scratch_mnist'}
    tc_zoo = {'CatsDogs':[0, 1], 'Imagenet':[16,17,18,19], 'MNIST':[4, 7]}

    for task in task_zoo:
        if task!='Imagenet':
            tan_flag_zoo = [True, False]
        else:
            tan_flag_zoo = [False]
        for tan_flag in tan_flag_zoo:
            model = model_zoo[task]
            tc = tc_zoo[task]
            if task == 'Imagenet':
                cam_method_zoo = ['fullcam']
            else:
                cam_method_zoo = ['gradcam', 'fullcam', 'gradcampp', 'xgradcam']
            # naturalimage_runner(target_category=None, model_flag=model, task=task, dataset_split='val',
            #                     max_iter=None, randomization=False, random_severity=0,
            #                     eval_flag='basic', tan_flag=tan_flag, cam_method=cam_method_zoo)
            for tc_s in tc:
                naturalimage_runner(target_category=tc_s, model_flag=model, task=task, dataset_split='val',
                                    max_iter=None, randomization=False, random_severity=0,
                                    eval_flag='corr_logit', tan_flag=tan_flag, cam_method=cam_method_zoo)
    # catsdog3d_runner(target_category=1, task='catsdogs3d', dataset_split='val')

    # medical_runner(target_category=1, task='luna', dataset_split='val')

    # esmira_runner(target_category=1, data_dir='D:\\ESMIRA\\ESMIRA_common',
    #             target_catename=['CSA'], target_site=['Wrist'], target_dirc=['TRA', 'COR'])

    # ramris_pred_runner(data_dir='', target_category=['EAC'], 
    #              target_site=['Wrist'], target_dirc=['TRA', 'COR'],
    #              target_biomarker=['SYN'],
    #              target_reader=['Reader1', 'Reader2'], task_mode='clip', phase='train',
    #              full_img=True)

    # list_of_output = [item for item in range(43)]
    # ramris_pred_runner(data_dir='D:\\ESMIRA\\ESMIRA_common', target_category=None, 
    #              target_site=['Wrist'], target_dirc=['TRA', 'COR'],
    #              target_biomarker=None,
    #              target_reader=['Reader1', 'Reader2'], task_mode='clip', phase='train',
    #              full_img=True, dimension=2,
    #              target_output=list_of_output)

    # list_of_output = [item for item in range(10)]
    # ramris_pred_runner(data_dir='D:\\ESMIRA\\ESMIRA_common', target_category=None, 
    #              target_site=['Wrist'], target_dirc=['TRA', 'COR'],
    #              target_biomarker=['TSY'],
    #              target_reader=['Reader1', 'Reader2'], task_mode='clip', phase='train',
    #              full_img=True, dimension=2,
    #              target_output=list_of_output,
    #              cluster=None, cluster_start=0)

    # list_of_output = [item for item in range(15)]
    # ramris_pred_runner(data_dir='D:\\ESMIRA\\ESMIRA_common', target_category=None, 
    #              target_site=['Wrist'], target_dirc=['TRA', 'COR'],
    #              target_biomarker=['BME'],
    #              target_reader=['Reader1', 'Reader2'], task_mode='clip', phase='train',
    #              full_img=True, dimension=2,
    #              target_output=list_of_output,
    #              cluster=None, cluster_start=0, tanh=False)