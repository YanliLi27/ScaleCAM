import cv2
import numpy as np
import torch
from cam_components.core.activations_and_gradients import ActivationsAndGradients
from cam_components.utils.svd_on_activations import get_2d_projection
from torch import nn
from scipy.special import softmax


class BaseCAM_A:
    def __init__(self,
                 model,
                 target_layers,
                 ram:bool,
                 use_cuda=False,
                 reshape_transform=None,
                 compute_input_gradient=False,
                 uses_gradients=True
                 ):
        self.model = model.eval()
        self.target_layers = target_layers
        self.ram = ram  # if num_out=1 --> regression tasks, others --> classification
        self.cuda = use_cuda
        if self.cuda:
            self.model = model.cuda()
        self.reshape_transform = reshape_transform
        self.compute_input_gradient = compute_input_gradient
        self.uses_gradients = uses_gradients
        self.activations_and_grads = ActivationsAndGradients(
            self.model, target_layers, reshape_transform)
        self.softamx = nn.Softmax(dim=1)

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
        for i in range(len(target_category)):
            loss = loss + output[i, target_category[i]]
        return loss

    def get_cam_image(self,
                      input_tensor,
                      target_layer,
                      target_category,
                      activations,
                      grads):
        # input_tensor -- for 2D: [batch, channel, y, x], for 3D: [batch, channel, z, y, x]
        weights = self.get_cam_weights(input_tensor, target_layer,
                                       target_category, activations, grads)
        # activations -- for 2D: [batch, channel, y, x], for 3D: [batch, channel, z, y, x]
        # print the max of weights and activations
        # print('max of the weights:', torch.max(weights))
        # print('max of the activations: ', torch.max(activations))
        if len(input_tensor.shape)==4:
            if len(weights.shape)==2:
                weighted_activations = weights[:, :, None, None] * activations
                # weighted_activations -- [batch, channel, width, length]
            elif len(weights.shape)==4:
                weighted_activations = weights * activations
            else:
                raise ValueError(f'the length {len(weights.shape)} of 3D weights is not valid')
            
            grad_cam = weighted_activations.sum(axis=1)  # from [batch, channel, length, width] to [batch, length, width]
            cam_grad_max_value = np.max(grad_cam, axis=(1, 2)).flatten()
            cam_grad_min_value = np.min(grad_cam, axis=(1, 2)).flatten()
            
            # channel_numbers = weighted_activations.shape[1]   # weighted_activations[0] = [channel, length, width] numpy array
            # B = weighted_activations.shape[0]
            cam = weighted_activations.sum(axis=(2, 3)) 
        elif len(input_tensor.shape)==5:
            if len(weights.shape)==2:
                weighted_activations = weights[:, :, None, None, None] * activations
                # weighted_activations -- [batch, channel, depth, width, length]
            elif len(weights.shape)==5:
                weighted_activations = weights * activations
                # weighted_activations -- [batch, channel, depth, width, length]
            else:
                print(f'the shape of input tensor:{input_tensor.shape}')
                raise ValueError(f'the length {len(weights.shape)} of 3D weights is not valid, should be 2--gradcam, or 5--fullcam.')

            grad_cam = weighted_activations.sum(axis=1) 
            # from [batch, channel, depth, length, width] to [batch, depth, length, width]
            cam_grad_max_value = np.max(grad_cam, axis=(1, 2, 3)).flatten()
            cam_grad_min_value = np.min(grad_cam, axis=(1, 2, 3)).flatten()
            
            # channel_numbers = weighted_activations.shape[1]   # weighted_activations[0] = [channel, depth, length, width] numpy array
            # B = weighted_activations.shape[0]
            cam = weighted_activations.sum(axis=(2, 3, 4))   # [channel]
        else:
            raise ValueError(f'the length {len(input_tensor.shape)} of 3D weights is not valid')

        return cam, cam_grad_max_value, cam_grad_min_value  # cam [batch, all_channels]


    def _score_calculation(self, output, batch_size, target_category=None):
        np_output = output.cpu().data.numpy()  # [batch*[2/1000]]
        if self.ram:
            if target_category is None:
                target_category = 0
            assert isinstance(target_category, int)
            prob_predict_category = np_output[:, target_category]  # 
            predict_category = target_category
            pred_scores = np_output[:, target_category]
            nega_scores = None
            target_category = [target_category] * batch_size
            return prob_predict_category, predict_category, pred_scores, nega_scores, target_category
        else:
            prob_predict_category = softmax(np_output, axis=-1)  # [batch*[2/1000 classes_normalized]]  # softmax进行平衡
            predict_category = np.argmax(prob_predict_category, axis=-1)  # [batch*[1]]  # 预测类取最大
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
            else:
                raise ValueError('not valid target_category')
        
        return prob_predict_category, predict_category, pred_scores, nega_scores, target_category


    def forward(self, input_tensor, target_category=None):
        if self.cuda:
            input_tensor = input_tensor.cuda()

        if self.compute_input_gradient:
            input_tensor = torch.autograd.Variable(input_tensor,
                                                   requires_grad=True)

        output = self.activations_and_grads(input_tensor)
        
        _, predict_category, pred_scores, _, target_category =\
            self._score_calculation(output, input_tensor.size(0), target_category)
            

        if self.uses_gradients:
            self.model.zero_grad()
            loss = self.get_loss(output, target_category)
            loss.backward(retain_graph=True)

        cam_per_layer, cam_grad_max_value, cam_grad_min_value = self.compute_cam_per_layer(input_tensor,
                                                                                           target_category)
        # list[target_layers,(array[batch, all_channel=512])]
        return cam_per_layer, predict_category, pred_scores, cam_grad_max_value, cam_grad_min_value  # 由于batch=1，target_layers=1， [1, 1, all_channels]

    def get_target_width_height(self, input_tensor):
        width, height = input_tensor.size(-1), input_tensor.size(-2)
        return width, height

    def compute_cam_per_layer(
            self,
            input_tensor,
            target_category):
        activations_list = [a.cpu().data.numpy()
                            for a in self.activations_and_grads.activations]
        grads_list = [g.cpu().data.numpy()
                      for g in self.activations_and_grads.gradients]
        # Loop over the saliency image from every layer
        cam_importance_matrix = []
        cam_max_matrix = []
        cam_min_matrix = []
        for target_layer, layer_activations, layer_grads in \
                zip(self.target_layers, activations_list, grads_list):
            cam, cam_grad_max_value, cam_grad_min_value = self.get_cam_image(input_tensor,
                                                                            target_layer,
                                                                            target_category,
                                                                            layer_activations,
                                                                            layer_grads) 
            # cam = [batch, all_channels]
            cam_importance_matrix.append(cam)  # list [target_layers, (array[batch, all_channels])]
            cam_max_matrix.append(cam_grad_max_value)
            cam_min_matrix.append(cam_grad_min_value)
        
        if len(self.target_layers)>1:
            return self._aggregate_multi_layers(cam_importance_matrix),\
                    self._aggregate_multi_layers(cam_max_matrix),\
                    self._aggregate_multi_layers(cam_min_matrix)
        else:
            return cam_importance_matrix[0], cam_max_matrix[0], cam_min_matrix[0]
        # list[target_layers,(array[batch, channel, length, width])]
        # to -  1(target_layers/batch) * all channels(256 for each - 2* 256))

    def _aggregate_multi_layers(self, cam_per_target_layer):  # 当target layer不止一层时采用平均的方式合成到一起
        cam_per_target_layer = np.concatenate(cam_per_target_layer, axis=1)  # axis=1是为了在list内越过batch那一维度
        # [target_layers* (array[batch, all_channels])] --> []
        return np.mean(cam_per_target_layer, axis=1)

    def __call__(self,
                 input_tensor,
                 target_category=None):

        return self.forward(input_tensor,
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
