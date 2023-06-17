from typing import Union
from cam_components.agent_main import CAMAgent
from torchvision.models.vgg import vgg16


if __name__ == '__main__':
    
    # information needed:
    model = vgg16()  # your model
    target_layer = model.features[-1]
    groups:int=1
    num_out_channel:int=512  # the channel of the target layer -- only needed when use analyzer
    # potential solution:
    # num_out_channel = target_layer.out_channels
    dataset = 'your dataset'
    num_classes:int=2
    fold_order:int=0

    # -------------------------------- optional: -------------------------------- #
    cam_method:str = 'gradcam'
    im_dir:str = './output/im/taskname'
    cam_dir:str = './output/cam/taskname'
    batch_size:int=16
    target_category:Union[None, int, str]=1  # info of the running process
    # more functions
    maxmin_flag:bool=False  # normalization
    remove_minus_flag:bool=True # remove the part below zero, default: True in the original Grad CAM
    im_selection_mode:str='all'  # use feature selection or not -- relied on the importance matrices
    im_selection_extra:float=0.05  # importance matrices attributes
    max_iter=None  # early stop
    randomization:bool=False
    random_severity:int=1  # model randomization for sanity check
    # -------------------------------- optional end -------------------------------- #

    Agent = CAMAgent(model, target_layer, dataset,  
                num_out_channel, num_classes,  
                groups, fold_order,  
                # optional:
                cam_method, im_dir, cam_dir, # cam method and im paths and cam output
                batch_size, target_category,  # info of the running process
                maxmin_flag, remove_minus_flag, # creator
                im_selection_mode, im_selection_extra, # importance matrices attributes
                max_iter,  # early stop
                randomization, random_severity  # model randomization for sanity check
                )
    Agent.analyzer_main()
    Agent.creator_main(eval_act='corr', mm_ratio=1.5, use_origin=True)