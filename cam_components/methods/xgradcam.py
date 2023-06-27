import numpy as np
from cam_components.core.base_cam_analyzer import BaseCAM_A


class XGradCAM_A(BaseCAM_A):
    def __init__(self, model, target_layers, ram, use_cuda=False,
                 reshape_transform=None,compute_input_gradient=False,
                 uses_gradients=True):
        super(
            XGradCAM_A,
            self).__init__(
            model,
            target_layers,
            ram, 
            use_cuda,
            reshape_transform,
            compute_input_gradient,
            uses_gradients)

    def get_cam_weights(self,
                        input_tensor,
                        target_layer,
                        target_category,
                        activations,
                        grads,
                        ):
        if len(input_tensor.shape)==4:
            sum_activations = np.sum(activations, axis=(2, 3))
            eps = 1e-7
            weights = grads * activations / \
                (sum_activations[:, :, None, None] + eps)
            weights = weights.sum(axis=(2, 3))
            return weights
        elif len(input_tensor.shape)==5:
            sum_activations = np.sum(activations, axis=(2, 3, 4))
            eps = 1e-7
            weights = grads * activations / \
                (sum_activations[:, :, None, None, None] + eps)
            weights = weights.sum(axis=(2, 3, 4))
            return weights
        else:
            raise ValueError(f'the shape is not supported: {input_tensor.shape}')


