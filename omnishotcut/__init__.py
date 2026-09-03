import logging
import os
import numpy as np
import torch
from huggingface_hub import hf_hub_download
from huggingface_hub.utils import enable_progress_bars

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    logger.addHandler(logging.StreamHandler())

from omnishotcut.engine import load_model, _run_on_numpy
from omnishotcut.datasets.utils import _decode_video, _resize_video
from omnishotcut.label_correspondence import unique_intra_label_mapping, intra_int2string, inter_int2string


_DEFAULT_HF_FILENAME = "OmniShotCut_ckpt.pth"


class OmniShotCutModel:

    def __init__(self, model, model_args):
        self._model = model
        self._model_args = model_args

    def inference(self, video, mode="clean_shot", overlap=10):
        """Run shot cut detection on a video.

        Args:
            video: str file path | np.ndarray (T,H,W,3) RGB | torch.Tensor (T,H,W,3) RGB.
                   Any input is resized to the model's process resolution
                   (process_width x process_height, e.g. 128x96) before inference,
                   so the input H/W can be arbitrary. Non-uint8 arrays/tensors are
                   treated as float in [0, 1].
            mode: "clean_shot" — general cuts only (no transitions)
                  "default"    — all detected shots with full labels
            overlap: number of overlap frames between adjacent inference windows

        Returns:
            ranges:       list of [start_frame, end_frame]
            intra_labels: list of int (0=General, 1=Dissolve, 2=Wipes, ...)
            inter_labels: list of int (0=New_Start, 1=Hard_Cut, 2=Transition_Source, ...)
        """
        if isinstance(video, str):
            video_np = _decode_video(video, self._model_args.process_width, self._model_args.process_height)
        elif isinstance(video, torch.Tensor):
            if video.ndim != 4 or video.shape[-1] != 3:
                raise ValueError(f"Tensor must be (T, H, W, 3), got {tuple(video.shape)}")
            video_np = video.cpu().numpy()
            if video_np.dtype != np.uint8:
                if video_np.min() < 0.0 or video_np.max() > 1.0:
                    raise ValueError(f"Float tensor must be in [0, 1], got range [{video_np.min():.3f}, {video_np.max():.3f}]")
                video_np = (video_np * 255).astype(np.uint8)
        else:
            video_np = np.asarray(video)
            if video_np.ndim != 4 or video_np.shape[-1] != 3:
                raise ValueError(f"numpy array must be (T, H, W, 3), got {video_np.shape}")
            if video_np.dtype != np.uint8:
                if video_np.min() < 0.0 or video_np.max() > 1.0:
                    raise ValueError(f"Float array must be in [0, 1], got range [{video_np.min():.3f}, {video_np.max():.3f}]")
                video_np = (video_np * 255).astype(np.uint8)

        video_np = _resize_video(video_np, self._model_args.process_width, self._model_args.process_height)

        ranges, intra_labels, inter_labels = _run_on_numpy(video_np, self._model, self._model_args, overlap)

        if mode == "clean_shot":
            general_idx = unique_intra_label_mapping["general"]
            keep = [i for i, lbl in enumerate(intra_labels) if lbl == general_idx]
            ranges = np.array(ranges)[keep].tolist() if keep else []
            return ranges

        intra_labels = [intra_int2string.get(x, str(x)) for x in intra_labels]
        inter_labels = [inter_int2string.get(x, str(x)) for x in inter_labels]
        return ranges, intra_labels, inter_labels


def _touch_download_counter(repo_id, filename="config.json"):
    """Make an HF model load count toward the repo's download statistics.

    The Hub counts downloads by requests to a repo's "query file" (config.json
    by default, when no library filter is registered), not to the weight file.
    Fetching only OmniShotCut_ckpt.pth therefore never increments the counter,
    so we also fetch config.json here. Best-effort: any failure is ignored.
    """
    try:
        hf_hub_download(repo_id=repo_id, filename=filename)
    except Exception as e:
        logger.debug(f"HF download-count file '{filename}' not fetched: {e}")


def load(checkpoint_path, filename=_DEFAULT_HF_FILENAME):
    """Load model weights and return an OmniShotCutModel instance.

    checkpoint_path can be:
      - a local file path  → load directly (filename ignored)
      - a HF repo ID       → download the specified filename from that repo
    """
    if not os.path.exists(checkpoint_path):
        logger.info(f"Downloading checkpoint from HuggingFace: {checkpoint_path} ...")
        enable_progress_bars()
        _touch_download_counter(checkpoint_path)   # count this load toward HF download stats
        checkpoint_path = hf_hub_download(repo_id=checkpoint_path, filename=filename)

    logger.info(f"Loading OmniShotCut from {checkpoint_path} ...")
    model, model_args = load_model(checkpoint_path)
    logger.info("OmniShotCut loaded successfully.")
    return OmniShotCutModel(model, model_args)
