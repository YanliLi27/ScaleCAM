from predefiend_runner import naturalimage_runner, catsdog3d_runner, esmira_runner, medical_runner


if __name__ == '__main__':
    # for test
    # naturalimage_runner(target_category=10, model_flag='vgg', task='Imagenet', dataset_split='val',
    #                     max_iter=None, randomization=False, random_severity=0)
    # catsdog3d_runner(target_category=1, task='catsdogs3d', dataset_split='val')
    # medical_runner(target_category=1, task='luna', dataset_split='val')
    esmira_runner(target_category=1, data_dir='D:\\ESMIRA\\ESMIRA_common',
                target_catename=['CSA'], target_site=['Wrist'], target_dirc=['TRA', 'COR'])