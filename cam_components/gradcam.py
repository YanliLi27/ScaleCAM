import numpy as np
from cam_components.core.base_cam_analyzer import BaseCAM_A


class GradCAM_A(BaseCAM_A):
    def __init__(self, model, target_layers, use_cuda=False,
                 reshape_transform=None,compute_input_gradient=False,
                 uses_gradients=True):
        super(
            GradCAM_A,
            self).__init__(
            model,
            target_layers,
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
            return np.mean(grads, axis=(2, 3)) # for 2D: [batch, channel, y, x], for 3D: [batch, channel, z, y, x]
        elif len(input_tensor.shape)==5:
            return np.mean(grads, axis=(2, 3, 4))
        else:
            raise ValueError(f'the shape is not supported: {input_tensor.shape}')


