# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
    Data Loader for OmniShotCut. Modified from DETR.
"""

import os, sys, shutil
import random
from typing import List, Union, Optional, Tuple
from pathlib import Path
import numpy as np
import ffmpeg
import torch
import torch.utils.data
import imageio
import torchvision
import torch.nn.functional as F
from torch.utils.data import Dataset
from PIL import Image


# Import files from the local folder
root_path = os.path.abspath('.')
sys.path.append(root_path)
from datasets.transforms import Video_Augmentation_Transform
from config.label_correspondence import unique_intra_label_mapping, unique_inter_label_mapping
from util.visualization import visualize_concated_frames



def align_segments_to_crop(segments, crop_start, crop_len):
    """
    segments: list of (start, end, intra_label, inter_label) in global frame indices, [start, end)
    returns:
        ab: list of (start2, end2) in cropped local indices
        ys: list of labels
    """
    s = int(crop_start)
    e = s + int(crop_len)

    ab = []         # Refer to the time range
    intras = []         # Refer to the label
    inters = []

    for start, end, intra, inter in segments:
        start = int(start)
        end = int(end)
        na = max(start, s)
        nb = min(end, e)
        if nb <= na:
            continue
        ab.append([na - s, nb - s])
        intras.append(intra)
        inters.append(inter)

    
    # Change the first Inter to be new start
    inters[0] = unique_inter_label_mapping["new_start"]
        

    return ab, intras, inters



def pad_to_length(x, N, pad_value=(1.0, 0.0)):
    K = x.shape[0]
    assert K <= N

    pad = torch.tensor(pad_value, dtype=x.dtype, device=x.device)
    pad = pad.unsqueeze(0).expand(N - K, 2)   # (N-K, 2)

    return torch.cat([x, pad], dim=0)




class CutAnything_Dataloader(Dataset):

    def __init__(self, args, set_type):


        # Fetch information
        self.set_type = set_type          # "train" or "val" set for the dataloader
        self.args = args
        self.process_height = args.process_height
        self.process_width = args.process_width
        self.max_process_window_length = args.max_process_window_length         # The max number of frames we need
        self.has_overlength_prob = args.has_overlength_prob                  # If we have overlength window
        self.max_padding_length = args.max_process_window_length - args.min_video_in_padding                # Max padding frames allowed in max_process_window_length
        self.num_queries = args.num_queries
        


        # Choose Data Info
        if set_type == "train":
            data_info_path = args.train_data_info_path
        elif set_type == "val":
            data_info_path = args.val_data_info_path
        if not os.path.exists(data_info_path):
            print("We cannot find", data_info_path)
            assert(os.path.exists(data_info_path))


        # Load pkl files
        data_info = []
        for sub_pkl_name in sorted(os.listdir(data_info_path)):
            sub_pkl_path = os.path.join(data_info_path, sub_pkl_name)
            data_info.extend(np.load(sub_pkl_path, allow_pickle=True))


        # Collect
        if set_type == "val" and args.max_val_num is not None:           #  None means to use all
            data_info = data_info[:args.max_val_num]            
        print("Total number of", set_type, "dataset is", len(data_info))
        self.data_info = data_info



        # Augmentation (Horizontal Flip + Color Jitter + Gray Scale + Blur) + Transform (ImageNet Normalization)
        if set_type == "train":
            self.video_transform = Video_Augmentation_Transform(
                                                                    set_type = "train",
                                                                    horizontal_flip_prob = 0.5,                 # Horizontal Flip
                                                                    vertical_flip_prob = 0.0,                   # Vertical Flip  
                                                                    jitter_prob = 0.15,                         # Color Jitter Prob 
                                                                    jitter_param = (0.05, 0.05, 0.05, 0.02),    # Color Jitter  
                                                                    grayscale_prob = 0.0,                       # GraryScale
                                                                    blur_prob = 0.03,                           # Blur 
                                                                    blur_kernel_size = 3,                       # Should be odd number
                                                                    blur_sigma = (0.1, 0.3),            
                                                                    noise_prob = 0.0,                           # Add Gaussian Noise
                                                                    noise_sigma = (0.003, 0.01),
                                                                    noise_clip = (0.0, 1.0),
                                                                    compression_prob = 0.05,                    # Image-based compression
                                                                    compression_choices = ["jpeg", "webp"],
                                                                )
        
        elif set_type == "val":
            self.video_transform = Video_Augmentation_Transform(
                                                                    set_type = "val"         
                                                                )

        else:
            raise NotImplementedError("we do not support set type of", set_type)
        

    def __len__(self):
        return len(self.data_info)


    def __getitem__(self, idx):
        
        while True: # Iterate until there is a valid video read
            try:

                # Fetch
                data_dict = self.data_info[idx]
                video_path = data_dict["video_path"]
                gt_ranges = data_dict["transition_ranges"]
                gt_intra_labels = data_dict["transition_intra_labels"]
                gt_inter_labels = data_dict["transition_inter_labels"]
                fps = data_dict["fps"]
                assert(len(gt_ranges) == len(gt_intra_labels) and len(gt_ranges) == len(gt_inter_labels))


                # Sanity Check
                if not os.path.exists(video_path):
                    print("We cannot find", video_path)
                    assert(os.path.exists(video_path))


                ############################################################ Construct the Video Inputs #########################################################################

                # Read the video by ffmpeg
                resolution = str(self.process_width) + "x" + str(self.process_height)
                video_stream, err = ffmpeg.input(
                                                    video_path
                                                ).output(
                                                    "pipe:", format = "rawvideo", pix_fmt = "rgb24", s = resolution, vsync = 'passthrough',
                                                ).run(
                                                    capture_stdout = True, capture_stderr = True
                                                )      # The resize is already included
                video_np_full = np.frombuffer(video_stream, np.uint8).reshape(-1, self.process_height, self.process_width, 3)
                original_num_frames = len(video_np_full)
                if original_num_frames < self.max_process_window_length:
                    print("We only has", original_num_frames, "number of frames!")
                    raise Exception("The number of frames in the video is too short")            # Exception Cases will choose a new idx
            
                # Visualize (Comment Out Later)
                # visualize_concated_frames(video_np_full, "instance_"+str(idx), gt_ranges, max_frames_per_img=400, end_range_exclusive=True)


                
                # Crop the video to be fixed length
                if self.set_type == "train" and random.random() < self.has_overlength_prob:       # Overlength case, might have padding
                    start_sample_frame_idx = random.randint(0, original_num_frames - self.max_process_window_length + self.max_padding_length - 1) 
                else:       # Regular Case (Must inside the full video)
                    start_sample_frame_idx = random.randint(0, original_num_frames - self.max_process_window_length - 1) if self.set_type == "train" else 0
                end_sample_frame_idx = min(len(video_np_full), start_sample_frame_idx + self.max_process_window_length)
                video_np = video_np_full[ start_sample_frame_idx : end_sample_frame_idx]       


                # Add padding
                num_padding_frames = self.max_process_window_length - len(video_np)
                assert(num_padding_frames <= self.max_padding_length)           
                black_padding_frames = np.zeros((num_padding_frames, self.process_height, self.process_width, 3), dtype=video_np.dtype)
                video_np = np.concatenate([video_np, black_padding_frames], axis=0)


                # Video Data Transform + Augmentation       
                video_tensor = self.video_transform(video_np, idx)                  # output shape is (F, C, H, W)

                ##################################################################################################################################################################





                ######################################################## Construct the Label System ##############################################################################
            
                # Construct the Standard Label: [End_Frame_Idx, Intra-Label, Inter-Label].   
                standard_labels = [[ *gt_ranges[clip_idx], unique_intra_label_mapping[gt_intra_labels[clip_idx]], unique_inter_label_mapping[gt_inter_labels[clip_idx]] ] for clip_idx in range(len(gt_ranges))]
                

                # Map the GT based on the start_sample_frame_idx position
                cropped_ranges, crop_intra_classification_label, crop_inter_classification_label = align_segments_to_crop(standard_labels, start_sample_frame_idx, self.max_process_window_length)


                ## Sanity Check: the cropped number of clips must be less than the number of query
                if len(cropped_ranges) > self.num_queries:
                    raise Exception("The number of clips ",  len(cropped_ranges), " is more than the number of query!")
                if len(cropped_ranges) != len(crop_intra_classification_label) or len(cropped_ranges) != len(crop_inter_classification_label):
                    raise Exception("We cannot find ranges to be aligned with labels!")


                # Prepare the GT Video Classification Label
                intra_label_tensor = torch.tensor(crop_intra_classification_label)    
                inter_label_tensor = torch.tensor(crop_inter_classification_label)    
                pad_len = self.num_queries - intra_label_tensor.numel()
                intra_label_tensor = F.pad(intra_label_tensor, (0, pad_len), "constant", unique_intra_label_mapping["padding"])
                inter_label_tensor = F.pad(inter_label_tensor, (0, pad_len), "constant", unique_inter_label_mapping["padding"])


                # Prepare the GT Shot Range
                shot_labels_tensor = torch.tensor(cropped_ranges)[:, 1].to(torch.int64)                   # [Inclusive, Exclusive) 
                shot_labels_tensor = F.pad(shot_labels_tensor, (0, pad_len), "constant", self.max_process_window_length)
                
                
                # Write as dictionary
                gt_target = {"shot_labels" : shot_labels_tensor, "intra_clip_labels" : intra_label_tensor, "inter_clip_labels" : inter_label_tensor}                  

                ############################################################################################################################################################


                # Build Auxiliary info for dictionary
                aux_info = {
                                "idx" : idx,
                                "video_path" : video_path,
                                "fps" : fps,
                                "start_frame_idx" : start_sample_frame_idx,
                                "end_frame_idx" : end_sample_frame_idx,
                            }


            except Exception as error:
                print("We face error", error, "and we will fetch next one!")
                old_idx = idx
                idx = random.randint(0, len(self.data_info))
                print("We cannot process the video", old_idx, " and we choose a new idx of ", idx)
                continue        # For any error occurs, we run it again with new idx proposed (a random int less than current value)

            break


        # Return
        return video_tensor, gt_target, aux_info
            



def build(args, set_type):

    dataset = CutAnything_Dataloader(args, set_type)
    return dataset
