'''
    This file is for OmniShotCut benchmark massive testing; Not used right now
'''
import os, sys, shutil
import argparse
import numpy as np
import math
import subprocess
import cv2
from tqdm import tqdm
import ffmpeg
import time
import torch
import json
import torchvision.transforms as T
from torch.utils.data import DataLoader
import pickle
import warnings
warnings.filterwarnings("ignore", category=UserWarning)


# Import files from the local folder
root_path = os.path.abspath('.')
sys.path.append(root_path)
from config.argument_setting import get_args_parser
from architecture.backbone import build_backbone
from architecture.transformer import build_transformer
from architecture.model import OmniShotCut
from datasets.transforms import Video_Augmentation_Transform
from util.visualization import visualize_concated_frames, concat_image_lists_horizontal
from config.label_correspondence import unique_intra_label_mapping, unique_inter_label_mapping
from test_code.inference import single_video_infernece, dump_list_of_dict
from evaluation.evaluate_SBD import evaluate_metrics

# Video Transform
video_transform = Video_Augmentation_Transform(set_type = "val")





def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
                            "--checkpoint_path",
                            type = str,
                            default = "/scratch/usy5km/Cut_Anything/cut_anything_checkpoints/results_training_v12/ckpt_epoch82.pth",
                            help = "Path to checkpoint file."
                        )
    parser.add_argument(
                            "--test_dataset_pkl_path",
                            type = str,
                            default = "/scratch/usy5km/Cut_Anything/cut_anything_benchmark/labels_5round.pkl",
                            help = "Path to test dataset pkl file."
                        )
    parser.add_argument(
                            "--result_store_path",
                            type = str,
                            default = "results.json",
                            help="Path to save result json."
                        )
    parser.add_argument(
                            "--num_context_frames",
                            type = int,
                            default = 0,
                            help = "Path to save result json."
                        )
    parser.add_argument(
                            "--visual_store_folder_path",
                            type = str,
                            default = None,
                            help = "Path to save visualization results. Set to None to disable."
                        )
    parser.add_argument(
                            "--merge_sudden_jump",
                            action = "store_true",
                            default = False,
                            help = "Whether to merge sudden jump."
                        )

    return parser.parse_args()




if __name__ == '__main__':

    # Setting
    inference_args = parse_args()
    checkpoint_path = inference_args.checkpoint_path
    test_dataset_pkl_path = inference_args.test_dataset_pkl_path
    result_store_path = inference_args.result_store_path
    visual_store_folder_path = inference_args.visual_store_folder_path
    merge_sudden_jump = inference_args.merge_sudden_jump
    


    # Prepare the folder
    if visual_store_folder_path is not None:
        os.makedirs(visual_store_folder_path, exist_ok = True)


    # Load Checkpoint & Model Config
    assert(os.path.exists(checkpoint_path))
    state_dict = torch.load(checkpoint_path, map_location='cpu')
    model_args = state_dict['args']
    print("Checkpoint stored args are", model_args)


    # Init the Model
    print("Load OmniShotCut Model!")
    backbone = build_backbone(model_args)
    transformer = build_transformer(model_args)
    model = OmniShotCut(
                            backbone,
                            transformer,
                            num_intra_relation_classes = model_args.num_intra_relation_classes,
                            num_inter_relation_classes = model_args.num_inter_relation_classes,
                            num_frames = model_args.max_process_window_length, 
                            num_queries = model_args.num_queries,
                            aux_loss = model_args.aux_loss,
                        )
    model.load_state_dict(state_dict['model'], strict=True)
    model.to("cuda")
    model.eval()
    


    # Read the pkl file
    with open(test_dataset_pkl_path, "rb") as f:
        test_data = pickle.load(f)



    # Iterate all cases
    print("Start Inference!")
    pred_results = []
    start_time = time.time()
    for instance_idx, info_dict in enumerate(tqdm(test_data, desc="Testing")):
        
        # Fetch info
        video_path = info_dict["video_path"]
        if not os.path.exists(video_path):
            print("We cannot find", video_path)
            assert(False)
        # print("video path is", video_path, "for instance", instance_idx)
        

        # Init result log
        pred_result = {}
        pred_result["video_path"] = video_path
        pred_result["gt_ranges"] = info_dict["ranges"]
        pred_result["gt_intra_labels"] = info_dict["intra_labels"]
        pred_result["gt_inter_labels"] = info_dict["inter_labels"]
        pred_result["gt_confidences"] = info_dict["confidences"]


        # Do the single inference
        pred_ranges_full, pred_intra_labels_full, pred_inter_labels_full, video_np_full = single_video_infernece(video_path, model, model_args, inference_args)


        # Append prediction resutls
        pred_result["pred_ranges"] = pred_ranges_full
        pred_result["pred_intra_labels"] = pred_intra_labels_full
        pred_result["pred_inter_labels"] = pred_inter_labels_full
        pred_results.append(pred_result)



        # Visualize
        if visual_store_folder_path is not None:

            # Visualize predictions
            prediction_visual_store_path = os.path.join(visual_store_folder_path, "instance" + str(instance_idx) + "_pred") 
            if os.path.exists(prediction_visual_store_path):
                shutil.rmtree(prediction_visual_store_path)
            pred_saved_paths = visualize_concated_frames(video_np_full, prediction_visual_store_path, pred_ranges_full, max_frames_per_img=264, end_range_exclusive=True, fps=fps, start_index = 0)


            # Visualize the GT results
            gt_visual_store_path = os.path.join(visual_store_folder_path, "instance" + str(instance_idx) + "_gt") 
            if os.path.exists(gt_visual_store_path):
                shutil.rmtree(gt_visual_store_path)
            gt_ranges_full = info_dict['ranges']
            gt_saved_paths = visualize_concated_frames(video_np_full, gt_visual_store_path, gt_ranges_full, max_frames_per_img=264, end_range_exclusive=True, fps=fps, start_index = 0)


            # Merge Pred and GT on One for easier visual
            merged_visual_store_path = os.path.join(visual_store_folder_path, "instance" + str(instance_idx) + "_merged") 
            if os.path.exists(merged_visual_store_path):
                shutil.rmtree(merged_visual_store_path)
            merged_paths = concat_image_lists_horizontal(           # Left: ours predictions；Right: GT
                                                            list1 = pred_saved_paths,
                                                            list2 = gt_saved_paths,
                                                            out_dir = merged_visual_store_path,
                                                            bar_width = 80,                     
                                                            bar_color = (0, 255, 0),            # (0, 255, 0) is green
                                                        )


    # Store the result as json
    dump_list_of_dict(pred_results, result_store_path)
    


    # Do the evluation here automatically
    evaluate_metrics(result_store_path, last_frame_exclusive=True)



    # Final Log
    print("Total time spent is", int(time.time() - start_time), "s!")
    print("Finished!")



    


