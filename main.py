from predefiend_runner import naturalimage_runner, catsdog3d_runner, esmira_runner, medical_runner


if __name__ == '__main__':
    # for test
    naturalimage_runner(target_category=None, model_flag='resnet', task='CatsDogs', dataset_split='val',
                        max_iter=None, randomization=False, random_severity=0)
    # catsdog3d_runner(target_category=None, model_flag='resnet', task='catsdogs3d', dataset_split='val')
    # medical_runner(target_category=None, task='ddsm', dataset_split='val')
    # esmira_runner(target_category=None, data_dir='D:\\ESMIRA\\ESMIRA_common',
    #             target_catename=['EAC','ATL'], target_site=['Wrist'], target_dirc=['TRA', 'COR'])