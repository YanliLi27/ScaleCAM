import numpy as np
from cam_components.core.base_cam_analyzer import BaseCAM_A


class FullCAM_A(BaseCAM_A):
    def __init__(self, model, target_layers, num_out, use_cuda=False,
                 reshape_transform=None,compute_input_gradient=False,
                 uses_gradients=True):
        super(
            FullCAM_A,
            self).__init__(
            model,
            target_layers,
            num_out,
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
        return grads


