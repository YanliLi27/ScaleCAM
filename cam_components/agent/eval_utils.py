import numpy as np
from torchvision import transforms
from scipy.special import softmax
import os


def cam_regularizer(mask):
    mask = np.maximum(mask, 0)
    mask = np.minimum(mask, 1)
    mask = mask - np.min(mask)/(np.max(mask)-np.min(mask)+1e-7)  # for dataset-level normalization
    return mask

def cam_input_normalization(cam_input):
    if cam_input.shape[1]==3:
        data_transform = transforms.Compose([
                                            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                            ])
        cam_input = data_transform(cam_input)
    return cam_input


def pred_score_calculator(input_size, output, target_category=None, origin_pred_category=None, out_logit:bool=True):
    np_output = output.cpu().data.numpy()
    prob_predict_category = softmax(np_output, axis=-1)  # [batch*[2\1000 classes_normalized]]
    if origin_pred_category is not None:
        predict_category = origin_pred_category
    else:
        predict_category = np.argmax(prob_predict_category, axis=-1)

    if target_category is None:
        target_category = predict_category
        if out_logit:
            pred_scores = np.max(np_output, axis=-1)
            nega_scores = np.sum(np_output, axis=-1)
        else:
            # pred_scores = np.max(prob_predict_category, axis=-1)
            arg = np.arange(0, prob_predict_category.shape[0])  # arg - batch_size[0, 1, ... , 16]
            pred_scores = prob_predict_category[arg, target_category]  # [batch, 1000] -> [batch]
            nega_scores = None
    elif type(target_category)==int:
        # assert(len(target_category) == input_size)
        
        if out_logit:
            matrix_zero = np.zeros([len(np_output), prob_predict_category.shape[-1]], dtype=np.int8)
            matrix_zero[:][target_category] = 1
            pred_scores = np.max(matrix_zero * np_output, axis=-1)
            nega_scores = np.sum(np_output, axis=-1)
        else:
            matrix_zero = np.zeros([len(np_output), prob_predict_category.shape[-1]], dtype=np.int8)  # TODO for parallel, change the 1/0 to batch size
            matrix_zero[:][target_category] = 1
            prob_predict_category = matrix_zero * prob_predict_category
            pred_scores = np.max(prob_predict_category, axis=-1)
            nega_scores = None
    
    elif target_category == 'GT':
        target_category = origin_pred_category.to('cpu').data.numpy().astype(int)
        matrix_zero = np.zeros([len(np_output), prob_predict_category.shape[-1]], dtype=np.int8)
        matrix_zero[list(range(len(np_output))), target_category] = 1
        pred_scores = np.max(matrix_zero* np_output, axis=-1)
        nega_scores = np.sum(np_output, axis=-1)
    else:
        raise TypeError(f'type of {target_category} is {type(target_category)}')
    return target_category, pred_scores, nega_scores  # both [batch_size, 1]


def text_save(save_path, increase, decrease, total_samples):
    if not os.path.isfile(save_path):
            with open(save_path, 'w') as F:
                F.write('Increase and decrease:')
    with open(save_path, 'a') as F:
        F.write('\n')
        F.write(f'total number: {str(total_samples)}')  
        F.write('\n')
        F.write(f'increase: {str(increase)}')
        F.write('\n')
        F.write(f'decrease: {str(decrease)}')


