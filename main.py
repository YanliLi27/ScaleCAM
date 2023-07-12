from predefiend_runner import naturalimage_runner, catsdog3d_runner, esmira_runner, medical_runner, ramris_pred_runner


if __name__ == '__main__':
    # for test
    naturalimage_runner(target_category=[0, 1], model_flag='vgg', task='CatsDogs', dataset_split='val',
                        max_iter=50, randomization=False, random_severity=0)
    # catsdog3d_runner(target_category=1, task='catsdogs3d', dataset_split='val')
    # medical_runner(target_category=1, task='luna', dataset_split='val')
    # esmira_runner(target_category=1, data_dir='D:\\ESMIRA\\ESMIRA_common',
    #             target_catename=['CSA'], target_site=['Wrist'], target_dirc=['TRA', 'COR'])
    # ramris_pred_runner(data_dir='', target_category=['EAC'], 
    #              target_site=['Wrist'], target_dirc=['TRA', 'COR'],
    #              target_biomarker=['SYN'],
    #              target_reader=['Reader1', 'Reader2'], task_mode='clip', phase='train',
    #              full_img=True)