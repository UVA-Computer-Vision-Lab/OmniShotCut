'''
    Util for collate function and related needs
'''
import os, sys, shutil
from typing import Optional, List
import torch
from torch import Tensor


# Import files from the local folder
root_path = os.path.abspath('.')
sys.path.append(root_path)
from util.misc import NestedTensor





def _max_by_axis(the_list):
    # type: (List[List[int]]) -> List[int]
    maxes = the_list[0]
    for sublist in the_list[1:]:
        for index, item in enumerate(sublist):
            maxes[index] = max(maxes[index], item)
    return maxes



def nested_tensor_from_tensor_list(tensor_list: List[Tensor], split=True):
    # Modified from VisTR, which shows a possible solution to handle video inputs

    # Split all video frames to one list, like an image form
    if split:
        # tensor_list = [tensor.split(3, dim=0) for tensor in tensor_list]            
        tensor_list = [item for sublist in tensor_list for item in sublist]           # The length of tensor_list equals to Batch Size * #Frames


    # Process each single one
    if tensor_list[0].ndim == 3:                # Expected (C, H, W) dimension

        # Same as DETR
        max_size = _max_by_axis([list(img.shape) for img in tensor_list])
        # min_size = tuple(min(s) for s in zip(*[img.shape for img in tensor_list]))
        batch_shape = [len(tensor_list)] + max_size
        b, c, h, w = batch_shape
        dtype = tensor_list[0].dtype
        device = tensor_list[0].device
        tensor = torch.zeros(batch_shape, dtype=dtype, device=device)
        mask = torch.ones((b, h, w), dtype=torch.bool, device=device)

        # Add Padding
        for img, pad_img, m in zip(tensor_list, tensor, mask):
            pad_img[: img.shape[0], : img.shape[1], : img.shape[2]].copy_(img)
            m[: img.shape[1], :img.shape[2]] = False

    else:

        raise ValueError('not supported')


    # Return Nested Tensor Form
    return NestedTensor(tensor, mask)           # tensor shape is (B*F, C, H, W) and mask shape is (B*F, H, W)



def collate_fn(batch):
    
    batch = list(zip(*batch))
    batch[0] = nested_tensor_from_tensor_list(batch[0])     # 0: Video Inputs;  1: GT Labels

    return tuple(batch)





