import cv2
import numpy as np
import torch
from cam_components.core.activations_and_gradients import ActivationsAndGradients
from cam_components.utils.svd_on_activations import get_2d_projection
from torch import nn
from scipy.special import softmax
import monai


class BaseCAM_P:
    def __init__(self,
                 model,
                 target_layers,
                 num_out:int,
                 importance_matrix,
                 use_cuda:bool=False,
                 groups:int=2,
                 reshape_transform=None,
                 compute_input_gradient:bool=False,
                 uses_gradients:bool=True,
                 value_max=None,
                 value_min=None,
                 remove_minus_flag:bool=False,
                 out_logit:bool=False,
                 tanh_flag:bool=False,
                 t_max:float=0.95,
                 t_min:float=0.05
                 ):
        if value_max:
            self.value_max = value_max
        else:
            self.value_max = None
        if value_min:
            self.value_min = value_min
        else:
            self.value_min = None
        self.model = model.eval()
        self.target_layers = target_layers
        self.num_out = num_out  # if num_out=1 --> regression tasks, others --> classification
        self.cuda = use_cuda
        if self.cuda:
            self.model = model.cuda()
        self.groups = groups
        self.reshape_transform = reshape_transform
        self.compute_input_gradient = compute_input_gradient
        self.im = importance_matrix  # [num_classes, channels]
        self.uses_gradients = uses_gradients
        self.remove_minus_flag = remove_minus_flag
        self.out_logit = out_logit
        self.activations_and_grads = ActivationsAndGradients(
            self.model, target_layers, reshape_transform)
        self.softamx = nn.Softmax(dim=1)

        self.tanh_flag = tanh_flag
        if self.tanh_flag and self.value_max:
            self.para_k = (np.arctanh(t_max) - np.arctanh(t_min))/(self.value_max-self.value_min)
            self.para_b = (np.arctanh(t_max)*self.value_min-np.arctanh(t_min)*self.value_max)/(self.value_min-self.value_max)
        else:
            self.para_k = None
            self.para_b = None


    """ Get a vector of weights for every channel in the target layer.
        Methods that return weights channels,
        will typically need to only implement this function. """

    def get_cam_weights(self,
                        input_tensor,
                        target_layers,
                        target_category,
                        activations,
                        grads,
                        ):
        raise Exception("Not Implemented")

    def get_loss(self, output, target_category):
        loss = 0
        if len(output.shape) == 1:  # for regression with only one output
            return output[target_category]
        for i in range(len(target_category)):  # for classification with different output
            loss = loss + output[i, target_category[i]]
        return loss

    def get_cam_image(self,
                      input_tensor,
                      target_layer,
                      target_category,
                      activations,
                      grads):
        weights = self.get_cam_weights(input_tensor, target_layer,
                                       target_category, activations, grads)
        

        if len(input_tensor.shape)==4:
            B, C, L, D = activations.shape
            if len(weights.shape)==2:
                weighted_activations = weights[:, :, None, None] * activations
                # weighted_activations -- [batch, channel, width, length]
            elif len(weights.shape)==4:
                weighted_activations = weights * activations
            else:
                raise ValueError(f'the length {len(weights.shape)} of 2D weights is not valid')
        elif len(input_tensor.shape)==5:
            B, C, H, L, D = activations.shape
            if len(weights.shape)==2:
                weighted_activations = weights[:, :, None, None, None] * activations
                # weighted_activations -- [batch, channel, depth, width, length]
            elif len(weights.shape)==5:
                weighted_activations = weights * activations
                # weighted_activations -- [batch, channel, depth, width, length]
            else:
                raise ValueError(f'the length {len(weights.shape)} of 3D weights is not valid')
        else:
            raise ValueError(f'the length {len(input_tensor.shape)} of 3D weights is not valid')

        im_weights = np.zeros([B, C])
        if self.im is not None:  # if self.im not exist, use the original
            im_weights[:] = self.im[target_category]  # self.im [num_classes, all_channels] - im_weights [batch_size, all_channels]
        # im_weights [batch_size, channels] 
            if len(input_tensor.shape)==4:
                weighted_activations = im_weights[:, :, None, None] * weighted_activations  # [batch, im-channel, None, None] * [batch, channel, length, width]
            elif len(input_tensor.shape)==5:
                weighted_activations = im_weights[:, :, None, None, None] * weighted_activations # [batch, im-channel, None, None, None] * [batch, channel, depth, length, width]
        channel_numbers = weighted_activations.shape[1]   # weighted_activations[0] = [channel, length, width] numpy array
        # print('channel_number:{}'.format(channel_numbers))
        channel_per_group = channel_numbers // self.groups
        # print('channel_per_group:{}'.format(channel_per_group))
        if len(input_tensor.shape)==4:
            [B, C, L, D] = weighted_activations.shape
            # print('B,C,L,D:{}'.format(weighted_activations.shape))
            target_type = weighted_activations.dtype
            cam = np.zeros([B, self.groups, L, D], dtype=target_type) 
            for j in range (B):
                for i in range(self.groups):
                    cam[j, i, :] = weighted_activations[j, i*channel_per_group:(i+1)*channel_per_group, :].sum(axis=0)
                    # print('max of the group:{}'.format(cam[j, i, :].max()))
            # cam:[batch, groups=2, length, width], while original: [batch, length, width]
        elif len(input_tensor.shape)==5:
            [B, C, H, L, D] = weighted_activations.shape
            # print('B,C,L,D:{}'.format(weighted_activations.shape))
            target_type = weighted_activations.dtype
            cam = np.zeros([B, self.groups, H, L, D], dtype=target_type) 
            for j in range (B):
                for i in range(self.groups):
                    cam[j, i, :] = weighted_activations[j, i*channel_per_group:(i+1)*channel_per_group, :].sum(axis=0) # channel_per_group to 1
        return cam  # group:[batch, groups=2, length, width], while original: [batch, length, width]
        # For 3D-- group:[batch, groups=2, depth, length, width], while original: [batch, groups, depth, length, width]


    def _score_calculation(self, output, batch_size, gt=None, target_category=None):
        np_output = output.cpu().data.numpy()  # [batch*[2/1000]]
        if self.num_out>1:
            prob_predict_category = softmax(np_output, axis=-1)  # [batch*[2/1000 classes_normalized]]
            predict_category = np.argmax(prob_predict_category, axis=-1)  # [batch*[1]]
            if target_category is None:
                target_category = predict_category
                if self.out_logit:
                    pred_scores = np.max(np_output, axis=-1)
                    nega_scores = np.sum(np_output, axis=-1)
                else:
                    pred_scores = np.max(prob_predict_category, axis=-1)
                    nega_scores = None
            elif isinstance(target_category, int):
                if self.out_logit:
                    pred_scores = np_output[:, target_category]  # [batch*[2/1000]] -> [batch*1]
                    nega_scores = np.sum(np_output, axis=-1) - pred_scores # [batch*2/1000] -> [batch*1] - [batch*1]
                else:
                    pred_scores = prob_predict_category[:, target_category]  # [batch*[2/1000]] -> [batch*1]
                    nega_scores = None
                target_category = [target_category] * batch_size
                assert(len(target_category) == batch_size)
            elif target_category == 'GT':
                target_category = gt.to('cpu').data.numpy().astype(int)
                matrix_zero = np.zeros([len(np_output), prob_predict_category.shape[-1]], dtype=np.int8)
                matrix_zero[list(range(len(np_output))), target_category] = 1
                pred_scores = np.max(matrix_zero* np_output, axis=-1)
                nega_scores = np.sum(np_output, axis=-1)
            else:
                raise ValueError('not valid target_category')
        else:
            prob_predict_category = np_output
            predict_category = np.argmax(np_output, axis=-1)
            pred_scores = np_output[:, predict_category]
            nega_scores = None
            target_category = predict_category
        
        return prob_predict_category, predict_category, pred_scores, nega_scores, target_category


    def forward(self, input_tensor, gt, target_category=None):
        if self.cuda:
            input_tensor = input_tensor.cuda()

        if self.compute_input_gradient:
            input_tensor = torch.autograd.Variable(input_tensor,
                                                   requires_grad=True)

        output = self.activations_and_grads(input_tensor)

        prob_predict_category, predict_category, pred_scores, nega_scores, target_category =\
            self._score_calculation(output, input_tensor.size(0), gt, target_category)
        
        if self.uses_gradients:
            self.model.zero_grad()
            loss = self.get_loss(output, target_category)
            loss.backward(retain_graph=True)

        cam_per_layer = self.compute_cam_per_layer(input_tensor,
                                                   target_category,
                                                   prob_weights=prob_predict_category)
        # list[target_layers*batch,(array[channel, length, width])]
        # print('cam_per_later returned to the outer item: {}'.format(np.squeeze(cam_per_layer).shape))  # (array[batch, channel, length, width]) squeeze to remove the list -- and get the [batch, channel, length, width]
        # batch1[2, 512, 512], batch2.....

        return cam_per_layer, predict_category, pred_scores, nega_scores  # 由于batch=1，target_layers=1， [1, 1, groups, length, width]


    def get_target_width_height(self, input_tensor):
        if len(input_tensor.shape)==4:
            width, height = input_tensor.size(-2), input_tensor.size(-1)
            return width, height
        elif len(input_tensor.shape)==5:
            depth, width, height = input_tensor.size(-3), input_tensor.size(-2), input_tensor.size(-1)
            return depth, width, height
        else:
            raise ValueError(f'NOT supported input_tensor shape {input_tensor.shape}')


    def compute_cam_per_layer(
            self,
            input_tensor,
            target_category,
            prob_weights=1.0):
        activations_list = [a.cpu().data.numpy()
                            for a in self.activations_and_grads.activations]
        grads_list = [g.cpu().data.numpy()
                      for g in self.activations_and_grads.gradients]
        target_size = self.get_target_width_height(input_tensor)

        cam_per_target_layer = []
        cam_per_target_layer_per_batch = []
        # Loop over the saliency image from every layer

        for target_layer, layer_activations, layer_grads in \
                zip(self.target_layers, activations_list, grads_list):
            cam = self.get_cam_image(input_tensor,
                                     target_layer,
                                     target_category,
                                     layer_activations,
                                     layer_grads)
            # print('max of cam:', np.max(cam))
            # print('min of cam:', np.min(cam))
            # print('the cam from get_cam_image, should be [batch, groups=2, length, width], truth is: {}'.format(cam.shape))
            # cam = [batch, groups, length, width] -- 3D = [batch, groups, depth, length, width]
            # cam = np.squeeze(cam, axis=0)  # (1, 2, 64, 64) only works when the batch=1
            if self.remove_minus_flag:
                cam = np.maximum(cam, 0)
            for cam_item in cam:  # from cam[batch, groups, depth, length, width] to [groups, depth, length, width]
                if self.tanh_flag and self.para_k:
                    scaled = self.tanh_scale_cam_image(cam_item, target_size, prob_weights)
                else:
                    scaled = self.scale_cam_image(cam_item, target_size, prob_weights)
                # 放缩不需要改变，不只是放缩比例，更重要的是进行了归一化
                # print('max of scaled cam:', np.max(np.asarray(scaled)))
                # print('min of scaled cam:', np.min(np.asarray(scaled)))
                # scaled [groups, length, width] / 3D scaled [groups, depth, length, width]
                cam_per_target_layer_per_batch.append(scaled)  # list[batch* [groups, depth, length, width]]
            cam_per_target_layer.append(cam_per_target_layer_per_batch)  # list[target_layers,(array[batch, groups, (depth),length, width])]
            # list[num_target_layer*[batch*array[groups, depth, length, width]]]
        if len(self.target_layers)>1:
            return self._aggregate_multi_layers(cam_per_target_layer)
        else:
            return cam_per_target_layer[0]
        # - list[(num_target_layer*batch)*array[groups, depth, length, width]]]
        # - 1*16(target_layers/batch) * array(2(groups), 512, 512) -- [16*array[2, 512, 512]]


    def _aggregate_multi_layers(self, cam_per_target_layer):  # 当target layer不止一层时采用平均的方式合成到一起
        # cam_per_target_layer -- [layers* [batch, groups, (depth), length, width]]
        cam_per_target_layer = np.concatenate(cam_per_target_layer, axis=(1, 2))  # axis=1是为了在list内越过batch那一维度
        # axis=(1,2) for skip batch and groups
        # [target_layers* (array[batch, all_channels])] --> []
        return np.mean(cam_per_target_layer, axis=(1, 2))


    def scale_cam_image(self, cam, target_size=None, prob_weights=1.0):
        result = []
        # cam 2d[groups, length, width]/3d[groups, depth, length, width]  
        # it's ok to have normalization inside each groups. -- but we need the prob_weights to fix it.
        for img in cam:  # [length, width]/[depth, length, width]
            # print('max of img:', np.max(img))
            # print('min of img:', np.min(img))
            if self.value_max and self.value_min:
                value_max = self.value_max
                if self.remove_minus_flag:
                    if value_max > 0:
                        value_min = 0
                    else:
                        value_min = value_min
                else:
                    value_min = self.value_min
            else:
                value_max = np.max(img) + 1e-7
                value_min = np.min(img)
            # print(value_max, ' and ', value_min)
            img = (img - value_min) / (value_max - value_min)
            # print('max of img:', np.max(img))
            # print('min of img:', np.min(img))
            # img = img * prob_weights  # 根据置信度进行修正
            if len(target_size) == 3: # 3d image
                # do not use 'linear' which is only for 3-d tensor (1d vectors)
                resize_fun = monai.transforms.Resize(target_size, mode='trilinear')  # receive image with shape c,z,y,x
                img = resize_fun(img[None,:,:,:])[0]  # z,y,x
                img = img.numpy()
            else:
                img = cv2.resize(img, target_size)  # for 2d image
            result.append(img)
        result = np.float32(result)

        return result
    

    def tanh_scale_cam_image(self, cam, target_size=None, prob_weights=1.0):
        result = []
        # cam 2d[groups, length, width]/3d[groups, depth, length, width]  
        # it's ok to have normalization inside each groups. -- but we need the prob_weights to fix it.
        for img in cam:  # [length, width]/[depth, length, width]
            img = np.tanh(self.para_k*img+self.para_b)
            # print('max of img:', np.max(img))
            # print('min of img:', np.min(img))
            # img = img * prob_weights  # 根据置信度进行修正
            if len(target_size) == 3: # 3d image
                # do not use 'linear' which is only for 3-d tensor (1d vectors)
                resize_fun = monai.transforms.Resize(target_size, mode='trilinear')  # receive image with shape c,z,y,x
                img = resize_fun(img[None,:,:,:])[0]  # z,y,x
                img = img.numpy()
            else:
                img = cv2.resize(img, target_size)  # for 2d image
            result.append(img)
        result = np.float32(result)
        return result


    def __call__(self,
                 input_tensor,
                 gt,
                 target_category=None):

        return self.forward(input_tensor,
                            gt,
                            target_category)  # [cam, predict_category]

    def __del__(self):
        self.activations_and_grads.release()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, exc_tb):
        self.activations_and_grads.release()
        if isinstance(exc_value, IndexError):
            # Handle IndexError here...
            print(
                f"An exception occurred in CAM with block: {exc_type}. Message: {exc_value}")
            return True
