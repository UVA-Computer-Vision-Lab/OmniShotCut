import os
import sys
import shutil
import argparse
import copy
import json
import numpy as np
import torch
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

from huggingface_hub import hf_hub_download
from huggingface_hub.utils import enable_progress_bars

from omnishotcut.engine import load_model, single_video_inference
from omnishotcut.label_correspondence import unique_intra_label_mapping, intra_int2string, inter_int2string
from omnishotcut.util.visualization import visualize_concated_frames

_HF_REPO = "uva-cv-lab/OmniShotCut_v1.5"
_HF_FILENAME = "OmniShotCut_ckpt.pth"


def dump_list_of_dict(data, save_path, indent=4):
    """
    Save list[dict] as JSON.
    Convert pred_intra_labels / pred_inter_labels from int to string.
    """

    def convert_item(item):
        item = copy.deepcopy(item)

        if "pred_intra_labels" in item:
            item["pred_intra_labels"] = [
                intra_int2string.get(x, f"Unknown_{x}")
                for x in item["pred_intra_labels"]
            ]

        if "pred_inter_labels" in item:
            item["pred_inter_labels"] = [
                inter_int2string.get(x, f"Unknown_{x}")
                for x in item["pred_inter_labels"]
            ]

        return item

    def format_dict(d, level):
        indent_str = " " * (indent * level)
        inner_indent = " " * (indent * (level + 1))

        lines = [indent_str + "{"]
        items = list(d.items())

        for i, (k, v) in enumerate(items):
            value_str = json.dumps(v, ensure_ascii=False)
            comma = "," if i < len(items) - 1 else ""
            lines.append(f'{inner_indent}"{k}": {value_str}{comma}')

        lines.append(f"{indent_str}}}")
        return "\n".join(lines)

    data = [convert_item(item) for item in data]

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
                            default = None,
                            help = "Path to local checkpoint file. Auto-downloads from HuggingFace if not set."
                        )
    parser.add_argument(
                            "--input_video_path",
                            type = str,
                            default = "__assets__/demo_video1.mp4",
                            help = "Path to the input video path."
                        )
    parser.add_argument(
                            "--result_store_path",
                            type = str,
                            default = "results.json",
                            help="Path to save result json."
                        )
    parser.add_argument(
                            "--overlap_window_length",
                            type = int,
                            default = 10,
                            help = "Number of overlapped frames between adjacent inference windows."
                        )
    parser.add_argument(
                            "--visual_store_folder_path",
                            type = str,
                            default = "demo_video_results",
                            help = "Path to save the visualization results. Set to None to disable."
                        )
    parser.add_argument(
                            "--mode",
                            type = str,
                            default = "default",
                            choices = ["default", "clean_shot"],
                            help = "Output Mode. 'default' means all Intra and Inter label. 'clean_shot' means only General Shot Cut without transitions. "
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
    if checkpoint_path is None:
        print(f"No checkpoint specified. Downloading from HuggingFace ({_HF_REPO})...")
        enable_progress_bars()
        from omnishotcut import _touch_download_counter
        _touch_download_counter(_HF_REPO)   # count this load toward HF download stats
        checkpoint_path = hf_hub_download(repo_id=_HF_REPO, filename=_HF_FILENAME)
    else:
        assert os.path.exists(checkpoint_path), f"Checkpoint not found: {checkpoint_path}"
    model, model_args = load_model(checkpoint_path)
    print("Checkpoint stored args are", model_args)


    # Do the inference
    print("Do the inference!")
    pred_ranges_full, pred_intra_labels_full, pred_inter_labels_full, video_np_full, fps = single_video_inference(
                                                                                                                    input_video_path,
                                                                                                                    model,
                                                                                                                    model_args,
                                                                                                                    inference_args.overlap_window_length,
                                                                                                                )



    # Visualize
    if visual_store_folder_path is not None:
        print("Visualize the results!")
        pred_saved_paths = visualize_concated_frames(
                                                        video_np_full,
                                                        visual_store_folder_path,
                                                        pred_ranges_full,
                                                        max_frames_per_img=264,
                                                        end_range_exclusive=True,
                                                        fps=fps,
                                                        start_index = 0,
                                                    )



    # For Clean Shot mode, we only leave general types
    if mode == "clean_shot":

        # Clean shot mode (No Transitions in the middle)
        general_type_idx = unique_intra_label_mapping["general"]
        effective_indices = []
        for idx, intra_label in enumerate(pred_intra_labels_full):
            if intra_label == general_type_idx:
                effective_indices.append(idx)

        # Reassign the shots
        pred_ranges_full = np.array(pred_ranges_full)[effective_indices].tolist()
        pred_intra_labels_full = np.array(pred_intra_labels_full)[effective_indices].tolist()
        pred_inter_labels_full = np.array(pred_inter_labels_full)[effective_indices].tolist()



    # Collect prediction results
    pred_result = {}
    pred_result["video_path"] = input_video_path
    pred_result["pred_ranges"] = pred_ranges_full
    pred_result["pred_intra_labels"] = pred_intra_labels_full
    pred_result["pred_inter_labels"] = pred_inter_labels_full

    # Dump to json
    dump_list_of_dict([pred_result], result_store_path)


    print("Finished!")
