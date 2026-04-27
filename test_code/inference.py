'''
    This file is to inference arbitrary video files for Shot Cut
'''
import os, sys, shutil
import argparse
import numpy as np
import math
import subprocess
import cv2
import ffmpeg
import json
import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader
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
from util.visualization import visualize_concated_frames
from config.label_correspondence import unique_intra_label_mapping, unique_inter_label_mapping


# Video Transform
video_transform = Video_Augmentation_Transform(set_type = "val")






def load_model(checkpoint_path: str):


    # Check the checkpoint
    checkpoint_path = os.path.abspath(checkpoint_path)
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    # if checkpoint_path in MODEL_CACHE:
    #     return MODEL_CACHE[checkpoint_path]


    # Load state dict
    state_dict = torch.load(checkpoint_path, map_location="cpu")
    if "args" not in state_dict or "model" not in state_dict:
        raise ValueError("Checkpoint must contain keys: 'args' and 'model'.")


    # Load the model
    model_args = state_dict["args"]
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
    model.load_state_dict(state_dict["model"], strict=True)
    model.to("cuda")
    model.eval()

    # MODEL_CACHE[checkpoint_path] = (model, model_args)
    return model, model_args



def get_video_fps_safe(video_path: str, default_fps: float = 24.0) -> float:
    try:
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        if fps is None or fps <= 1e-6 or math.isnan(fps):
            return default_fps
        return float(fps)
    except Exception:
        return default_fps



def split_videos(video, chunk_size, num_context_frames):

    assert video.ndim == 4, "video must be (T, H, W, C)"
    total_num_frames, H, W, C = video.shape
    

    # Padding at the beginning
    black = np.zeros((num_context_frames, H, W, C), dtype=video.dtype)
    video = np.concatenate([black, video], axis=0)


    # Split Video to clips
    stride = chunk_size - 2 * num_context_frames
    cur_frame_idx = 0
    return_list = []
    while cur_frame_idx < total_num_frames:
        
        # Fetch the range
        cropped_videos = video[cur_frame_idx : cur_frame_idx + chunk_size]

        # Add padding if needed
        clip_num_adding_frames = chunk_size - len(cropped_videos)
        if clip_num_adding_frames > 0:
            black = np.zeros((clip_num_adding_frames, H, W, C), dtype=video.dtype)
            cropped_videos = np.concatenate([cropped_videos, black], axis=0)

        # Append all return info: (video_np, clip padding frames, global start frame idx)
        return_list.append([cropped_videos, clip_num_adding_frames])

        # Update
        cur_frame_idx += stride
        
    return return_list
    



def prune_non_context_ranges(pred_ranges, pred_intra_labels, pred_inter_labels, inference_window_size, num_context_frames):

    # Init
    new_pred_ranges, new_pred_intra_labels, new_pred_inter_labels = [], [], []


    # Iterate
    for shot_idx in range(len(pred_ranges)):

        # Fetch
        start_frame_idx, end_frame_idx = pred_ranges[shot_idx]

        # Check if we should skip
        if end_frame_idx <= num_context_frames:     # Beginning
            continue
        if start_frame_idx >= inference_window_size - num_context_frames:       # Ending
            break

        # Re align start & end
        aligned_start_frame_idx = max(start_frame_idx, num_context_frames) - num_context_frames
        aligned_end_frame_idx = min(end_frame_idx, inference_window_size - num_context_frames) - num_context_frames         # exclusive on the right range

        # Append
        new_pred_ranges.append([aligned_start_frame_idx, aligned_end_frame_idx])
        new_pred_intra_labels.append(pred_intra_labels[shot_idx])
        new_pred_inter_labels.append(pred_inter_labels[shot_idx])

    return new_pred_ranges, new_pred_intra_labels, new_pred_inter_labels




def merge_ranges(pred_ranges_full, pred_intra_labels_full, pred_inter_labels_full, pred_ranges, pred_intra_labels, pred_inter_labels):

    # Prepare
    last_frame_idx = pred_ranges_full[-1][-1] if len(pred_intra_labels_full) != 0 else 0


    # Merge last one of the list list 
    if len(pred_intra_labels_full) != 0 and pred_intra_labels_full[-1] == pred_intra_labels[0] and pred_inter_labels[0] == unique_inter_label_mapping['new_start']:  
        pred_ranges_full[-1][-1] = last_frame_idx + pred_ranges[0][-1]
        
        # Crop the first one
        pred_ranges = pred_ranges[1:]
        pred_intra_labels = pred_intra_labels[1:]
        pred_inter_labels = pred_inter_labels[1:]


    # Extend the following list
    for idx in range(len(pred_ranges)):
        start_frame_idx, end_frame_idx = pred_ranges[idx]

        pred_ranges_full.append([last_frame_idx + start_frame_idx, last_frame_idx + end_frame_idx])
        pred_intra_labels_full.append(pred_intra_labels[idx])
        pred_inter_labels_full.append(pred_inter_labels[idx])


    return pred_ranges_full, pred_intra_labels_full, pred_inter_labels_full
    





def single_video_inference(video_path, model, model_args, num_context_frames):


    # Init the parameter
    num_context_frames = num_context_frames
    max_process_window_length = model_args.max_process_window_length
    process_height, process_width = model_args.process_height, model_args.process_width


    # Read the Video
    fps = get_video_fps_safe(video_path)       # get_fps sometimes might have the bug
    video_stream, err = ffmpeg.input(
                                        video_path
                                    ).output(
                                        "pipe:", format = "rawvideo", pix_fmt = "rgb24", s = str(process_width) + "x" + str(process_height),  vsync = 'passthrough',
                                    ).run(
                                        capture_stdout = True, capture_stderr = True
                                    )      # The resize is already included
    video_np_full = np.frombuffer(video_stream, np.uint8).reshape(-1, process_height, process_width, 3)
    

    # Iterate all the clips
    pred_ranges_full, pred_intra_labels_full, pred_inter_labels_full = [], [], []
    for clip_idx, (video_np, num_pad_frames) in enumerate(split_videos(video_np_full, max_process_window_length, num_context_frames)):

        
        # Transform
        video_tensor = video_transform(video_np).unsqueeze(0).to("cuda")


        # Inference
        with torch.inference_mode():
            outputs = model(video_tensor)
        

        # Choose the label with max value
        probas_intra = outputs['intra_clip_logits'].softmax(-1)[0, :, :-1] 
        probas_inter = outputs['inter_clip_logits'].softmax(-1)[0, :, :-1]  
        range_probas = outputs['pred_shot_logits'].softmax(-1)[0, :, :-1]  
        query_intra_idx = probas_intra.argmax(dim=-1)
        query_inter_idx = probas_inter.argmax(dim=-1)
        query_range_idx = range_probas.argmax(dim=-1)


        # Print Prediction Results
        # print(f"\nPrediction Results for clip {clip_idx}:")
        pred_ranges, pred_intra_labels, pred_inter_labels = [], [], []
        start_frame_idx = 0
        for keep_idx in range(len(query_intra_idx)):

            # Fetch Label
            pred_intra_label = int(query_intra_idx[keep_idx].detach().cpu())
            pred_inter_label = int(query_inter_idx[keep_idx].detach().cpu())

            # Convert ranges from [0, 1] to video duration scales
            end_frame_idx = int(query_range_idx[keep_idx].detach().cpu())
            pred_range = [start_frame_idx, end_frame_idx]
            if start_frame_idx >= end_frame_idx:         # End the iteration
                continue
            # print("\tRange is", pred_range, "&& Intra + Inter Label is", pred_intra_label, pred_inter_label)             # NOTE: np.round() is the accurate way to write
            

            # Append the result 
            pred_ranges.append(pred_range)
            pred_intra_labels.append(pred_intra_label)
            pred_inter_labels.append(pred_inter_label)
            start_frame_idx = end_frame_idx
        
            
            # End
            if end_frame_idx >= max_process_window_length - num_pad_frames:
                break                # Touch the end / padding frames, we can jump out earlier
        

        # Prune predictions to the current range
        pred_ranges, pred_intra_labels, pred_inter_labels = prune_non_context_ranges(pred_ranges, pred_intra_labels, pred_inter_labels, max_process_window_length, num_context_frames)


        # Merge predicted results
        pred_ranges_full, pred_intra_labels_full, pred_inter_labels_full = merge_ranges(pred_ranges_full, pred_intra_labels_full, pred_inter_labels_full, pred_ranges, pred_intra_labels, pred_inter_labels)


    return pred_ranges_full, pred_intra_labels_full, pred_inter_labels_full, video_np_full, fps




def dump_list_of_dict(data, save_path, indent=4):
    """
    Save list[dict] as JSON
    """

    def format_dict(d, level):
        indent_str = " " * (indent * level)
        inner_indent = " " * (indent * (level + 1))

        lines = ["{"]
        items = list(d.items())

        for i, (k, v) in enumerate(items):
            value_str = json.dumps(v, ensure_ascii=False)
            comma = "," if i < len(items) - 1 else ""
            lines.append(f'{inner_indent}"{k}": {value_str}{comma}')

        lines.append(f"{indent_str}}}")
        return "\n".join(lines)

    with open(save_path, "w", encoding="utf-8") as f:
        f.write("[\n")

        for i, item in enumerate(data):
            dict_str = format_dict(item, level=1)
            comma = "," if i < len(data) - 1 else ""
            f.write(dict_str + comma + "\n")

        f.write("]\n")







def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
                            "--checkpoint_path",
                            type = str,
                            default = "checkpoints/OmniShotCut_ckpt.pth",
                            help = "Path to checkpoint file."
                        )
    parser.add_argument(
                            "--input_video_path",
                            type = str,
                            default = "/scratch/usy5km/Cut_Anything/examples/genshin_video.mp4",
                            help = "Path to the input video path."
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
                            default = "demo_video_results",
                            help = "Path to save visualization results. Set to None to disable."
                        )
    parser.add_argument(
                            "--mode",
                            type = str,
                            default = "default",
                            help = "Output Mode. default means all Intra and Inter label. Clean_Shot means only Shot Cut without transition and sudden jump. "
                        )

    return parser.parse_args()




if __name__ == '__main__':

    # Setting
    inference_args = parse_args()
    checkpoint_path = inference_args.checkpoint_path
    input_video_path = inference_args.input_video_path
    assert(os.path.exists(input_video_path))
    result_store_path = inference_args.result_store_path
    visual_store_folder_path = inference_args.visual_store_folder_path
    mode = inference_args.mode
    


    # Prepare the folder
    if visual_store_folder_path is not None:
        if os.path.exists(visual_store_folder_path):
            shutil.rmtree(visual_store_folder_path)
        os.makedirs(visual_store_folder_path)


    # Load Checkpoint & Model Config
    assert(os.path.exists(checkpoint_path))
    state_dict = torch.load(checkpoint_path, map_location='cpu')
    model_args = state_dict['args']
    print("Checkpoint stored args are", model_args)


    # Init the Model
    print("Loading OmniShotCut Model!")
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




    # Do the inference
    print("Do the inference!")
    pred_ranges_full, pred_intra_labels_full, pred_inter_labels_full, video_np_full, fps = single_video_inference(input_video_path, model, model_args, inference_args.num_context_frames)



    # Collect prediction resutls
    pred_result = {}
    pred_result["video_path"] = input_video_path
    pred_result["pred_ranges"] = pred_ranges_full
    pred_result["pred_intra_labels"] = pred_intra_labels_full
    pred_result["pred_inter_labels"] = pred_inter_labels_full



    # Visualize
    if visual_store_folder_path is not None:
        print("Visualize the results!")
        pred_saved_paths = visualize_concated_frames(video_np_full, visual_store_folder_path, pred_ranges_full, max_frames_per_img=264, end_range_exclusive=True, fps=24, start_index = 0)



    # Store the result as json
    if mode == "default":
        dump_list_of_dict([pred_result], result_store_path)
    elif mode == "clean_shot":
        # TODO: only leave the clean shots
        breakpoint()
        # dump_list_of_dict(pred_result, result_store_path, mode)
    else:
        raise NotImplementedError
    
    print("Finished!")