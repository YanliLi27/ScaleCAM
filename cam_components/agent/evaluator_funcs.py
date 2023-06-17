import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from scipy.special import softmax
from torchvision import transforms


def cam_regularizer(mask):
    mask = np.maximum(mask, 0)
    mask = np.minimum(mask, 1)
    return mask

def cam_input_normalization(cam_input):
    data_transform = transforms.Compose([
                                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                        ])
    cam_input = data_transform(cam_input)
    return cam_input


def pred_score_calculator(input_size, output, target_category=None, origin_pred_category=None):
    np_output = output.cpu().data.numpy()
    prob_predict_category = softmax(np_output, axis=-1)  # [batch*[2\1000 classes_normalized]]
    if origin_pred_category is not None:
        predict_category = origin_pred_category
    else:
        predict_category = np.argmax(prob_predict_category, axis=-1)

    if target_category is None:
        target_category = predict_category  # index batch_size[x, x, x, ...]
        arg = np.arange(0, prob_predict_category.shape[0])  # arg - batch_size[0, 1, ... , 16]
        pred_scores = prob_predict_category[arg, target_category]  # [batch, 1000] -> [batch]
    else:
        assert(len(target_category) == input_size)
        matrix_zero = np.zeros([1, prob_predict_category.shape[-1]], dtype=np.int8)
        matrix_zero[0][target_category] = 1
        prob_predict_category = matrix_zero * prob_predict_category
        pred_scores = np.max(prob_predict_category, axis=-1)
    return target_category, pred_scores  # both [batch_size, 1]


def cam_evaluator_step(cam_algorithm, model, target_layer, target_dataset,  # required attributes
                       general_args,
                       im=None, data_max_value=None, data_min_value=None, remove_minus_flag:bool=True,
                       max_iter=None,
                       device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
    test_loader = DataLoader(target_dataset, batch_size=general_args.batch_size, shuffle=False,
                                num_workers=1, pin_memory=True)
    model.eval()
    increase = 0.0
    drop = 0.0
    counter = 0

    for x,y in tqdm(test_loader):
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
            # list([batch, organ_groups, length, width]) -- 16* [1, 256, 256] - batch is in the list dimension
            
            grayscale_cam = np.array(grayscale_cam)
            grayscale_cam = cam_regularizer(np.array(grayscale_cam)) # -- [16, 1, 256, 256]
        # grayscale_cam:numpy [batch, groups, length, width] from 0 to 1, x:tensor [batch, in_channel, length, width] from low to high
        extended_cam = np.zeros(x.shape, dtype=np.float32)
        channel_per_group = x.shape[1] // general_args.groups
        for gc in range(general_args.groups):
            extended_cam[:, gc*channel_per_group:(gc+1)*channel_per_group, :] = np.expand_dims(grayscale_cam[:, gc, :], axis=1)
        # extended_cam: numpy [batch, in_channel, length, width]
        cam_input = torch.from_numpy(extended_cam).to(device) * x
        cam_input = cam_input_normalization(cam_input)
        cam_pred = model(cam_input)

        origin_category, single_origin_confidence = predict_category, confidence
        cam_category, single_cam_confidence = pred_score_calculator(x.shape[0], cam_pred, general_args.target_category_flag,
                                                                    origin_pred_category=origin_category)

        counter += x.shape[0]
        single_increase = single_origin_confidence < single_cam_confidence
        increase += single_increase.sum().item()
        single_drop = torch.relu(torch.from_numpy(single_origin_confidence - single_cam_confidence)).div(torch.from_numpy(single_origin_confidence) + 1e-7)
        drop += single_drop.sum().item()
        

        if max_iter is not None:
            if counter >= max_iter:
                print('counter meet max iter')
                break
    
    print('total samples:', counter)
    avg_increase = increase/counter
    avg_drop = drop/counter

    print('increase:', avg_increase)
    print('avg_drop:', avg_drop)