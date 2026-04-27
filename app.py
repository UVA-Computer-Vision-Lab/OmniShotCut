import os, sys, shutil
import json
import glob
import math
from pathlib import Path
from typing import Any, Dict, List, Tuple
import cv2
import ffmpeg
import numpy as np
import torch
import tempfile
import spaces

# Temp file bug of gradio
BASE_TMP_DIR = os.path.abspath("./gradio_tmp")
os.makedirs(BASE_TMP_DIR, exist_ok=True)
os.environ["TMPDIR"] = BASE_TMP_DIR
os.environ["TEMP"] = BASE_TMP_DIR
os.environ["TMP"] = BASE_TMP_DIR
os.environ["GRADIO_TEMP_DIR"] = BASE_TMP_DIR
tempfile.tempdir = BASE_TMP_DIR
import gradio as gr


# Import your existing project code
root_path = os.path.abspath(".")
sys.path.append(root_path)
from architecture.backbone import build_backbone
from architecture.transformer import build_transformer
from architecture.model import OmniShotCut
from datasets.transforms import Video_Augmentation_Transform
from util.visualization import visualize_concated_frames
from config.label_correspondence import unique_intra_label_mapping, unique_inter_label_mapping
from test_code.inference import single_video_inference, load_model



# -------------------------
# Global cache / constants
# -------------------------
video_transform = Video_Augmentation_Transform(set_type="val")
INTRA_ID2NAME = {v: k for k, v in unique_intra_label_mapping.items()}
INTER_ID2NAME = {v: k for k, v in unique_inter_label_mapping.items()}


# Fixed demo config
DEFAULT_CHECKPOINT_PATH = "checkpoints/OmniShotCut_ckpt.pth"
DEFAULT_NUM_CONTEXT_FRAMES = 0
DEFAULT_MAX_FRAMES_PER_IMG = 132
VIS_DIR = "demo_video_results"

# Public URL safe setting
MAX_GALLERY_PAGES = 20


# Prepare the checkpoint
if not os.path.exists(DEFAULT_CHECKPOINT_PATH):
    os.makedirs("checkpoints", exist_ok=True)
    os.system("wget -P checkpoints https://huggingface.co/uva-cv-lab/OmniShotCut/resolve/main/OmniShotCut_ckpt.pth")


# Load the model
checkpoint_path = DEFAULT_CHECKPOINT_PATH
model, model_args = load_model(checkpoint_path)



######################## Gallery Prepare ########################

def escape_html(x):
    x = "" if x is None else str(x)

    return (
            x.replace("&", "&amp;")
             .replace("<", "&lt;")
             .replace(">", "&gt;")
             .replace('"', "&quot;")
             .replace("'", "&#39;")
        )


def prepare_gallery(page_paths: List[str]):
    gallery_items = []

    for page_idx, page_path in enumerate(page_paths):
        gallery_items.append((page_path, f"Page {page_idx}"))

    return gallery_items


def prepare_result_table(
        pred_ranges: List[List[int]],
        pred_intra_labels: List[int],
        pred_inter_labels: List[int],
        fps: float,
    ) -> str:

    headers = [
                "Index",
                "Start Frame",
                "End Frame",
                "Start Time (s)",
                "End Time (s)",
                "Intra Label",
                "Inter Label",
            ]

    html = """
    <div class="result-table-wrap">
        <table class="result-table">
            <thead>
                <tr>
    """

    for h in headers:
        html += f"<th>{escape_html(h)}</th>"

    html += """
                </tr>
            </thead>
            <tbody>
    """

    for idx, pred_range in enumerate(pred_ranges):
        start_frame = int(pred_range[0])
        end_frame = int(pred_range[1])

        intra_id = int(pred_intra_labels[idx]) if idx < len(pred_intra_labels) else -1
        inter_id = int(pred_inter_labels[idx]) if idx < len(pred_inter_labels) else -1

        row = [
                idx,
                start_frame,
                end_frame,
                round(start_frame / fps, 3) if fps and fps > 0 else "",
                round(end_frame / fps, 3) if fps and fps > 0 else "",
                INTRA_ID2NAME.get(intra_id, str(intra_id)),
                INTER_ID2NAME.get(inter_id, str(inter_id)),
            ]

        html += "<tr>"
        for item in row:
            html += f"<td>{escape_html(item)}</td>"
        html += "</tr>"

    html += """
            </tbody>
        </table>
    </div>
    """

    return html


def list_sample_videos(asset_dir: str = "__assets__", max_samples: int = 8) -> List[List[str]]:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    asset_dir = os.path.join(script_dir, asset_dir)

    if not os.path.isdir(asset_dir):
        return []

    mp4_paths = []
    for name in sorted(os.listdir(asset_dir)):
        path = os.path.join(asset_dir, name)
        if os.path.isfile(path) and name.lower().endswith(".mp4"):
            mp4_paths.append([path])

    return mp4_paths[:max_samples]

sample_videos = list_sample_videos("__assets__", max_samples = 16)




@spaces.GPU(duration=120)
def run_demo(video_file):

    if video_file is None:
        raise gr.Error("Please upload a video first.")

    video_path = video_file if isinstance(video_file, str) else video_file.name
    if not os.path.exists(video_path):
        raise gr.Error(f"Video file does not exist: {video_path}")

    # Read the setting
    num_context_frames = DEFAULT_NUM_CONTEXT_FRAMES
    max_frames_per_img = DEFAULT_MAX_FRAMES_PER_IMG


    print("Start processing the video", video_path)
    pred_ranges, pred_intra_labels, pred_inter_labels, video_np_full, fps = single_video_inference(
                                                                                                    video_path = video_path,
                                                                                                    model = model,
                                                                                                    model_args = model_args,
                                                                                                    num_context_frames = int(num_context_frames),
                                                                                                )
    print("Finish running the video")
    
    # Prepare the folder
    if os.path.exists(VIS_DIR):
        shutil.rmtree(VIS_DIR)
    os.makedirs(VIS_DIR)

    # Visualize and store (Must Do!)
    page_paths = visualize_concated_frames(
                                            frames = video_np_full,
                                            out_dir = VIS_DIR,
                                            highlight_ranges_closed = pred_ranges,
                                            max_frames_per_img = int(max_frames_per_img),
                                            end_range_exclusive = True,
                                            fps = fps,
                                            start_index = 0,
                                        )

    gallery_paths = page_paths[:MAX_GALLERY_PAGES]

    result_table = prepare_result_table(
                                        pred_ranges = pred_ranges,
                                        pred_intra_labels = pred_intra_labels,
                                        pred_inter_labels = pred_inter_labels,
                                        fps = fps,
                                    )

    print("Visualization pages:", len(page_paths))
    print("Shown visualization pages:", len(gallery_paths))
    print("Predicted shots:", len(pred_ranges))

    return gr.update(value = prepare_gallery(gallery_paths)), gr.update(value = result_table)


def clear_demo_outputs():
    return gr.update(value = []), gr.update(value = "")



# -------------------------
# UI Design
# -------------------------
custom_css = """
#visual_gallery img {
    object-fit: contain !important;
}

#visual_gallery .thumbnail-item {
    object-fit: contain !important;
}

#visual_gallery .grid-wrap {
    align-items: start !important;
}

.result-table-wrap {
    width: 100%;
    max-height: 360px;
    overflow: auto;
    border: 1px solid #e5e7eb;
    border-radius: 10px;
}

.result-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 14px;
}

.result-table th {
    position: sticky;
    top: 0;
    background: #f9fafb;
    border-bottom: 1px solid #e5e7eb;
    padding: 8px 10px;
    text-align: left;
    white-space: nowrap;
}

.result-table td {
    border-bottom: 1px solid #f1f5f9;
    padding: 8px 10px;
    white-space: nowrap;
}

.result-table tr:hover {
    background: #f9fafb;
}
"""


with gr.Blocks(title="OmniShotCut Demo", css = custom_css) as demo:

    # Head title
    gr.Markdown(
                """
                <div align="center">

                # OmniShotCut: Shot-Query-based Video Transformer for Shot Boundary Detection

                **A sensitive and more informative SoTA shot boundary detection model.**

                <p style="white-space: nowrap;">
                    <a href="https://arxiv.org/abs/2505.21491"><img src="https://img.shields.io/badge/arXiv-Paper-b31b1b?logo=arxiv&logoColor=white"></a>
                    <a href="https://uva-computer-vision-lab.github.io/Frame-In-N-Out/"><img src="https://img.shields.io/badge/Project-Website-pink?logo=googlechrome&logoColor=white"></a>
                    <a href="https://huggingface.co/spaces/HikariDawn/FrameINO"><img src="https://img.shields.io/static/v1?label=%F0%9F%A4%97%20HF%20Space&message=Online+Demo&color=orange"></a>
                </p>

                </div>

                ---

                Upload a video and click **Run Inference**.
                """
            )

    with gr.Row():
        with gr.Column(scale=1):
            video_input = gr.Video(label = "Input Video", height = 480)
            run_button = gr.Button("Run Inference", variant="primary")

        with gr.Column(scale=1):
            gr.Markdown("## Visualization")
            gallery = gr.Gallery(
                                    label = None,
                                    columns = 1,
                                    height = 760,
                                    preview = True,
                                    elem_id = "visual_gallery",
                                    object_fit = "contain",
                                )

    gr.Markdown("## Predicted Shot Results")
    result_table = gr.HTML(
                            value = "",
                            elem_id = "result_table",
                        )

    
    gr.Markdown("## Sample Videos")
    gr.Examples(
                    examples = sample_videos,
                    inputs = [video_input],
                    label = "Choose a sample video",
                )


    run_button.click(
                        fn = clear_demo_outputs,
                        inputs = [],
                        outputs = [gallery, result_table],
                    ).then(
                        fn = run_demo,
                        inputs  =[video_input],
                        outputs = [gallery, result_table],
                    )



if __name__ == "__main__":
    demo.launch(share=True)