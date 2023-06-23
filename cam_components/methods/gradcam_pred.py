import numpy as np
from cam_components.core.base_cam_predictor import BaseCAM_P


class GradCAM_P(BaseCAM_P):
    def __init__(self, model, target_layers, num_out, importance_matrix, use_cuda=False, groups=1,
                 reshape_transform=None, compute_input_gradient=False, uses_gradients=True,
                 value_max=None, value_min=None, remove_minus_flag=False, out_logit=False,
                 tanh_flag:bool=False, t_max:float=0.95, t_min:float=0.05):
        super(
            GradCAM_P,
            self).__init__(
            model,
            target_layers,
            num_out,
            importance_matrix,
            use_cuda,
            groups,
            reshape_transform,
            compute_input_gradient,
            uses_gradients,
            value_max=value_max,
            value_min=value_min,
            remove_minus_flag=remove_minus_flag,
            out_logit=out_logit,
            tanh_flag=tanh_flag,
            t_max=t_max,
            t_min=t_min)

    def get_cam_weights(self,
                        input_tensor,
                        target_layer,
                        target_category,
                        activations,
                        grads,
                        ):
        if len(input_tensor.shape)==4:
            return np.mean(grads, axis=(2, 3)) # for 2D: [batch, channel, y, x], for 3D: [batch, channel, z, y, x]
        elif len(input_tensor.shape)==5:
            return np.mean(grads, axis=(2, 3, 4))
        else:
            raise ValueError(f'the shape is not supported: {input_tensor.shape}')


