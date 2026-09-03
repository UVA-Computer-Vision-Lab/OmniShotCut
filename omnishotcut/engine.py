'''
    This file is to inference arbitrary video files for Shot Cut
'''
import os, sys, shutil
import argparse
import numpy as np
import copy
import json
import torch
import warnings
warnings.filterwarnings("ignore", category=UserWarning)


# Import files from the local folder
root_path = os.path.abspath('.')
sys.path.append(root_path)
from omnishotcut.architecture.backbone import build_backbone
from omnishotcut.architecture.transformer import build_transformer
from omnishotcut.architecture.model import OmniShotCut
from omnishotcut.datasets.transforms import Video_Augmentation_Transform
from omnishotcut.datasets.utils import _decode_video, _video_fps
from omnishotcut.util.visualization import visualize_concated_frames
from omnishotcut.label_correspondence import unique_intra_label_mapping, unique_inter_label_mapping, intra_int2string, inter_int2string


# Video Transform
video_transform = Video_Augmentation_Transform(set_type = "val")


# Label ids for the opening shot of a video (general shot, new_start relation)
GENERAL_INTRA_ID = unique_intra_label_mapping["general"]
NEW_START_INTER_ID = unique_inter_label_mapping["new_start"]




def load_model(checkpoint_path: str):


    # Check the checkpoint
    checkpoint_path = os.path.abspath(checkpoint_path)
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")


    # Load state dict
    state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
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


    return model, model_args



def split_videos(video, chunk_size, overlap_size):

    assert video.ndim == 4, "video must be (T, H, W, C)"
    assert overlap_size >= 0 and overlap_size < chunk_size

    T, H, W, C = video.shape
    stride = chunk_size - overlap_size

    # Form the return list
    return_list = []
    window_start_idx = 0

    while window_start_idx < T:

        window_end_idx = window_start_idx + chunk_size
        valid_len = min(chunk_size, T - window_start_idx)

        # Fetch current window
        chunk = video[window_start_idx:min(window_end_idx, T)]

        # Padding
        num_pad_frames = chunk_size - valid_len
        if num_pad_frames > 0:
            black = np.zeros((num_pad_frames, H, W, C), dtype=video.dtype)
            chunk = np.concatenate([chunk, black], axis=0)

        # Valid region for this window. We split the overlap region by half.
        left_overlap = overlap_size // 2
        right_overlap = overlap_size - left_overlap

        if window_start_idx == 0:
            valid_start_idx = 0
        else:
            valid_start_idx = window_start_idx + left_overlap

        if window_end_idx >= T:
            valid_end_idx = T
        else:
            valid_end_idx = window_end_idx - right_overlap

        return_list.append(
            [
                chunk,
                num_pad_frames,
                window_start_idx,
                valid_start_idx,
                valid_end_idx,
                valid_len,
            ]
        )

        # End
        if window_end_idx >= T:
            break

        window_start_idx += stride

    return return_list



def decode_window_segments(query_intra_idx, query_inter_idx, query_range_idx, valid_len):
    '''Turn one window's query outputs into contiguous LOCAL segments.

    Returns a list of (start_local, end_local, intra_label, inter_label). Each
    segment's (intra, inter) is the model's prediction for the shot that BEGINS at
    start_local (start relation), which is the convention used by the ground truth.
    '''

    segments = []
    start_frame_idx = 0

    for keep_idx in range(len(query_intra_idx)):

        pred_intra_label = int(query_intra_idx[keep_idx].detach().cpu())
        pred_inter_label = int(query_inter_idx[keep_idx].detach().cpu())

        # Local end frame prediction, never allowed to enter the padding region
        end_frame_idx = int(query_range_idx[keep_idx].detach().cpu())
        end_frame_idx = min(end_frame_idx, valid_len)

        # Degenerate / duplicate query output, skip it
        if start_frame_idx >= end_frame_idx:
            continue

        segments.append((start_frame_idx, end_frame_idx, pred_intra_label, pred_inter_label))

        start_frame_idx = end_frame_idx

        # Reached the window end / padding, stop
        if end_frame_idx >= valid_len:
            break

    return segments



def collect_cuts_from_window(segments, window_start_idx, valid_start_idx, valid_end_idx):
    '''Emit GLOBAL cut points from one window's local segments.

    A cut is the START of a shot other than the window's opening shot. The very first
    segment (local frame 0) is skipped, because from this window's local view it is
    always a New_Start; the neighboring window that overlaps this position will have
    the same boundary as an interior segment start and will emit it with the correct
    label. Each cut carries the intra / inter label of the shot it begins.
    '''

    cuts = []
    for j in range(1, len(segments)):
        start_local = segments[j][0]                    # == segments[j - 1] end
        cut_pos_global = window_start_idx + start_local

        # Keep only cuts inside this window's valid region (tiles with neighbors)
        if valid_start_idx < cut_pos_global <= valid_end_idx:
            cuts.append(
                {
                    "pos": int(cut_pos_global),
                    "intra_label": int(segments[j][2]),
                    "inter_label": int(segments[j][3]),
                }
            )

    return cuts



def assemble_ranges(global_cuts, total_frames, duplicate_tolerance=0):
    '''Build the full-video shot ranges + labels from the collected global cuts.

    Ranges partition [0, total_frames]. The first range is the opening shot
    (general / new_start). Every later range starts at a cut and inherits that
    cut's label.
    '''

    # Sort and dedupe near-duplicate cuts coming from overlapping windows
    global_cuts = sorted(global_cuts, key=lambda c: c["pos"])
    deduped = []
    for c in global_cuts:
        if len(deduped) != 0 and abs(c["pos"] - deduped[-1]["pos"]) <= duplicate_tolerance:
            continue
        if c["pos"] <= 0 or c["pos"] >= total_frames:
            continue
        deduped.append(c)

    pred_ranges, pred_intra_labels, pred_inter_labels = [], [], []
    prev = 0

    for k, c in enumerate(deduped):
        pos = c["pos"]
        if pos <= prev:
            continue

        pred_ranges.append([int(prev), int(pos)])
        if k == 0:
            # Opening shot of the whole video
            pred_intra_labels.append(int(GENERAL_INTRA_ID))
            pred_inter_labels.append(int(NEW_START_INTER_ID))
        else:
            pred_intra_labels.append(int(deduped[k - 1]["intra_label"]))
            pred_inter_labels.append(int(deduped[k - 1]["inter_label"]))
        prev = pos

    # Final tail range [last_cut, total_frames]
    if prev < total_frames:
        pred_ranges.append([int(prev), int(total_frames)])
        if len(deduped) == 0:
            # Whole video is a single shot
            pred_intra_labels.append(int(GENERAL_INTRA_ID))
            pred_inter_labels.append(int(NEW_START_INTER_ID))
        else:
            pred_intra_labels.append(int(deduped[-1]["intra_label"]))
            pred_inter_labels.append(int(deduped[-1]["inter_label"]))

    return pred_ranges, pred_intra_labels, pred_inter_labels



def _windows_to_ranges(video_np, model, model_args, overlap_window_length):
    '''Shared core: run the model over overlapping windows and stitch the results
    into full-video ranges + labels. A cut is labeled by the shot that BEGINS it
    (start relation), and each window's opening segment is skipped, so interior
    cuts are no longer mislabeled New_Start. Localization is unchanged.
    '''

    max_process_window_length = model_args.max_process_window_length
    total_frames = len(video_np)

    global_cuts = []

    for clip_idx, (video_chunk, num_pad_frames, window_start_idx, valid_start_idx, valid_end_idx, valid_len) in enumerate(split_videos(video_np, max_process_window_length, overlap_window_length)):

        # Transform
        video_tensor = video_transform(video_chunk).unsqueeze(0).to("cuda")

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

        # Local contiguous segments, then global cuts (skips the window's opening segment)
        segments = decode_window_segments(query_intra_idx, query_inter_idx, query_range_idx, valid_len)
        global_cuts.extend(collect_cuts_from_window(segments, window_start_idx, valid_start_idx, valid_end_idx))

    return assemble_ranges(global_cuts, total_frames)



def single_video_inference(video_path, model, model_args, overlap_window_length):


    # Init the parameter
    process_height, process_width = model_args.process_height, model_args.process_width


    # Read the Video
    video_np_full = _decode_video(video_path, process_width, process_height)
    fps = _video_fps(video_path)


    # Windowed inference + stitching
    pred_ranges_full, pred_intra_labels_full, pred_inter_labels_full = _windows_to_ranges(
                                                                                            video_np_full,
                                                                                            model,
                                                                                            model_args,
                                                                                            overlap_window_length,
                                                                                        )


    return pred_ranges_full, pred_intra_labels_full, pred_inter_labels_full, video_np_full, fps



def _run_on_numpy(video_np, model, model_args, overlap_window_length):
    """Run inference on a pre-loaded numpy array (T, H, W, 3).
    Returns (ranges, intra_labels, inter_labels).
    """
    return _windows_to_ranges(video_np, model, model_args, overlap_window_length)
