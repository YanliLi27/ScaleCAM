import torch
import cv2
import os
from torch.utils.data import Dataset, DataLoader
import torchvision
from tqdm import tqdm
from main_func.main_components.utils.img_creator import origin_creator, cam_creator, origin_creator3d, cam_creator3d
import SimpleITK as sitk
import numpy as np


def cam_predictor_step(cam_algorithm, model, target_layer, target_dataset, cam_dir,  # required attributes
                       general_args,
                       im=None, data_max_value=None, data_min_value=None, remove_minus_flag:bool=True,
                       max_iter=None, set_mode:bool=True, use_origin:bool=True,
                       device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
    if set_mode:
        test_loader = DataLoader(target_dataset, batch_size=general_args.batch_size, shuffle=False,
                                    num_workers=1, pin_memory=True)
        in_fold_counter = 0

        model = model.to(device=device)
        model.eval()
        for x,y in tqdm(test_loader):
            if len(x.shape)==4: # x -- [batch, channel, y, x]
                origin_img = origin_creator(x, organ_groups=general_args.groups)
                # origin_img -- [batch, organ_groups, channel=3, y, x]
            elif len(x.shape)==5: # x -- [batch, channel, z, y, x]/[batch, group/channel=2, Depth, Length, Width]
                origin_img = origin_creator3d(x, organ_groups=general_args.groups)
                origin_img = origin_img.transpose(0, 1, 5, 2, 3, 4)  # because we got no way for colored 3d images
                # origin_img -- [batch, organ_groups, channel=3, z, y, x]
            x = x.to(dtype=torch.float32).to(device)
            y = y.to(dtype=torch.float32).to(device)
            
            with cam_algorithm(model=model,
                                target_layers=target_layer,
                                importance_matrix=im,  # im -- [batch, organ_groups * channels] - [batch, 2 * N]
                                use_cuda=True,
                                groups=general_args.groups,
                                value_max=data_max_value,
                                value_min=data_min_value,
                                remove_minus_flag=remove_minus_flag,
                                out_logit=False,
                                ) as cam:

                grayscale_cam, predict_category, confidence, nega_score = cam(input_tensor=x,
                                                                              gt=y,
                                                                              target_category=general_args.target_category_flag)
                # theory: grayscale_cam -- target_layer*batch * array[groups, (depth), length, width]
                # proved: grayscale_cam -- 16 * [1, 256, 256] - batch * [1, 256, 256]
                if not os.path.exists(cam_dir):
                    os.makedirs(cam_dir)
                # print(np.max(grayscale_cam[0]))
                # print(np.min(grayscale_cam[0]))

                for i in range(general_args.batch_size):
                    if len(grayscale_cam[i].shape) == 3:
                        concat_img_all, output_label, cf_num = cam_creator(grayscale_cam[i], predict_category[i],\
                                                                        confidence[i], general_args.groups, origin_img[i], \
                                                                        use_origin=use_origin)
                        # save the cam
                        str_labels = (str(y.data.cpu().numpy()[i]))[:2]
                        save_name = os.path.join(cam_dir, f'fold{general_args.fold_order}_tr{str_labels}pr{output_label}_{in_fold_counter}_cf{cf_num}.jpg')
                        in_fold_counter += 1
                        cv2.imwrite(save_name, concat_img_all)
                    elif len(grayscale_cam[i].shape) == 4:
                        # concat_img_all, origin_img_all, output_label, cf_num = cam_creator3d(grayscale_cam[i], predict_category[i],\
                        #                                                 confidence[i], general_args.groups, origin_img[i], \
                        #                                                 use_origin=use_origin)   
                        # print(grayscale_cam[i].shape)   
                        output_label = predict_category[i]
                        cf_num = str(np.around(confidence[i], decimals=3))
                        # save the cam
                        str_labels = (str(y.data.cpu().numpy()[i]))[:2]
                        save_name = os.path.join(cam_dir, f'fold{general_args.fold_order}_tr{str_labels}pr{output_label}_{in_fold_counter}_cf{cf_num}.nii.gz')
                        origin_save_name = save_name.replace('.nii.gz', '_ori.nii.gz')
                        in_fold_counter += 1
       
                        # TODO from [batch, organ_groups, z, y, x, channel] to [batch, organ_groups, channel, z, y, x]
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

                if max_iter:
                    if in_fold_counter >= max_iter:
                        break
    else:
        print('input is not a dataset, but something else')
        model = model.to(device=device)
        model.eval()
        x = torch.unsqueeze(target_dataset, dim=0) # from [C, L, D] to [1, C, L, D]
        origin_img = origin_creator(x, organ_groups=general_args.groups)
        x = x.to(dtype=torch.float32).to(device)
        with cam_algorithm(model=model,
                                target_layers=target_layer,
                                importance_matrix=im,  # im -- [batch, organ_groups * channels] - [batch, 2 * N]
                                use_cuda=True,
                                groups=general_args.groups,
                                value_max=data_max_value,
                                value_min=data_min_value,
                                remove_minus_flag=remove_minus_flag
                                ) as cam:
                grayscale_cam, predict_category, confidence = cam(input_tensor=x, 
                                                                target_category=general_args.target_category_flag)
                # TODO 3d update
                for i in range(general_args.batch_size):
                    concat_img_all, output_label, cf_num = cam_creator(grayscale_cam[i], predict_category[i], confidence[i], general_args.groups, origin_img[i])
                    if not os.path.exists(cam_dir):
                        os.makedirs(cam_dir)
                    save_name = os.path.join(cam_dir, f'example.jpg')
                    in_fold_counter += 1
                    cv2.imwrite(save_name, concat_img_all)