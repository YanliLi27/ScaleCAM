from typing import Union
from cam_components.agent_main import CAMAgent
from torchvision.models.vgg import vgg16


if __name__ == '__main__':
    
    # information needed:
    model = 'your model'
    groups:int=1 # your group convolution
    target_layer = 'your targeted layers'
    num_out_channel:int=512  # the channel of the target layer -- only needed when use analyzer
    # TODO potential solution:
    # num_out_channel = target_layer.out_channels
    dataset = 'your dataset'
    num_classes:int=2  # the label number in your dataset
    fold_order:int=0  # for cross-validation

    # if use scaling:
    # -------------------------------- optional: -------------------------------- #
    cam_method:str = 'gradcam'
    im_dir:str = './output/im/taskname'
    cam_dir:str = './output/cam/taskname'  # default in this 
    batch_size:int=16
    target_category:Union[None, int, str]=1  # info of the running process
    
    # more functions
    maxmin_flag:bool=False  # normalization
    remove_minus_flag:bool=True # remove the part below zero, default: True in the original Grad CAM
    im_selection_mode:str='all'  # use feature selection or not -- relied on the importance matrices -- default to be 'all' used
    im_selection_extra:float=0.05  # importance matrices attributes
    max_iter=None  # early stop -- for test
    randomization:bool=False
    random_severity:int=1  # model randomization for sanity check
    ram:bool=False # for regression tasks, supported for multi-output
    # -------------------------------- optional end -------------------------------- #

    Agent = CAMAgent(model, target_layer, dataset,  
                num_out_channel, num_classes,  
                groups, fold_order, ram,  
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