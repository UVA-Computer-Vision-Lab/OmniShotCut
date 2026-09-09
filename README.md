

<p align="center">
    <img src="__assets__/logo.png" height="100">
</p>

## OmniShotCut: Holistic Relational Shot Boundary Detection with Shot-Query Transformer

OmniShotCut is a sensitive and more informative SoTA for Shot Boundary Detection. \
OmniShotCut can detect shot changes of the video in diverse sources (anime, vlog, game, shorts, sports, screen recording, etc.), and recognize Sudden Jump and Transitions (dissolve, fade, wipe, etc.) by proposing a Shot-Query-based Video Transformer.


[![Paper](https://img.shields.io/badge/arXiv-Paper-b31b1b?logo=arxiv&logoColor=white)](https://arxiv.org/abs/2604.24762)
[![Website](https://img.shields.io/badge/Project-Website-pink?logo=googlechrome&logoColor=white)](https://uva-computer-vision-lab.github.io/OmniShotCut_website/)
<a href="https://huggingface.co/spaces/uva-cv-lab/OmniShotCut"><img src="https://img.shields.io/static/v1?label=%F0%9F%A4%97%20HF%20Space&message=Online+Demo&color=orange"></a>
<a href="https://huggingface.co/uva-cv-lab/OmniShotCut_v1.5"><img src="https://img.shields.io/static/v1?label=%F0%9F%A4%97%20HuggingFace&message=Model+Weight&color=orange"></a>


🔥 [Update](#Update) **|** 🐍 [Quick Start](#quick_start)  **|** 🔧 [Installation](#Installation) **|** ⚡ [Inference](#fast_inference) **|** 🤗 [Model Zoo](#model_zoo)




## <a name="Update"></a>Update 🔥🔥🔥
- [x] Release ArXiv paper
- [x] Release the inference weights
- [x] Release Gradio demo (with online)
- [x] Release 'pip install omnishotcut' version
- [x] **Released the v1.5 weight** — a larger model with fewer false cuts on continuous camera motion / lighting changes and better dense fast-cut recall. 
- [ ] Release the benchmark
- [ ] Release the training code and curation
      
:star: **If you like OmniShotCut, please help ⭐⭐star⭐⭐ this repo. Thanks!** :hugs:




<p align="center">
    <img src="__assets__/teaser.png" style="border-radius: 15px">
</p>

<p align="center">
    <img src="__assets__/model_architecture.png" style="border-radius: 15px">
</p>





## <a name="quick_start"></a> Quick Start 🐍

First install PyTorch with CUDA support, and make sure a working `ffmpeg` binary is
on your `PATH` (used for video decoding; e.g. `conda install -c conda-forge ffmpeg`).
Then install OmniShotCut:
```shell
pip install git+https://github.com/UVA-Computer-Vision-Lab/OmniShotCut.git
```

Once installed, running shot boundary detection is just a few lines:

```python
import omnishotcut

# Load model — accepts a local checkpoint path or HuggingFace repo
cut_model = omnishotcut.load("uva-cv-lab/OmniShotCut_v1.5", filename = "OmniShotCut_ckpt.pth")

# Run on a video file
ranges = cut_model.inference("video.mp4", mode="clean_shot")
```

`ranges` is a list of `[start_frame, end_frame]` pairs for each detected shot.
By default `mode="clean_shot"` returns only clean cuts (no transitions). 
Use `mode="default"` to also get dissolves, wipes, and fades with their labels:

```python
ranges, intra_labels, inter_labels = cut_model.inference("video.mp4", mode="default")
```

Besides video file paths, `inference()` also accepts **numpy** arrays and **torch** tensors directly — both `(T, H, W, 3)` RGB (either `uint8`, or float in `[0, 1]`). The input H/W can be arbitrary; frames are resized to the model's process resolution automatically, so results match the video-file path:

```python
ranges = cut_model.inference(frames_thwc, mode="clean_shot")  # frames_thwc: (T, H, W, 3)
```



## <a name="Installation"></a> Full Local Installation 🔧
```shell
conda create -n OmniShotCut python=3.10
conda activate OmniShotCut
conda install -c conda-forge ffmpeg   # required: frame-accurate video decoding backend
pip install -r requirements.txt
pip install -e .
```

> **Note:** video decoding uses `ffmpeg` (via `ffmpeg-python`), so a working `ffmpeg`
> binary must be on your `PATH`. Installing it from `conda-forge` as above is the
> most reliable way to get one with all shared libraries present.


## <a name="fast_inference"></a> Gradio Demo ⚡⚡⚡
Local Gradio can be created by simply running the following:
```shell
python app.py 
```
Click "Running on **public** URL".




## <a name="inference"></a> Inference ⚡

This section presents a more formal and controllable way to run inference.

First, let us download the checkpoint
```shell
mkdir checkpoints
cd checkpoints
wget https://huggingface.co/uva-cv-lab/OmniShotCut_v1.5/resolve/main/OmniShotCut_ckpt.pth
```

We provide several inference modes. The 'default' mode outputs the intra and inter labels we define.
However, we believe that most users might want the most direct results, which is the general shots without any transitions. 
To this end, please use '--mode clean_shot'.

Execute the inference by:
```shell
python inference.py  --checkpoint_path checkpoints/OmniShotCut_ckpt.pth  --input_video_path __assets__/demo_video1.mp4  --overlap_window_length 10  --mode default
```

Results are saved to `results.json`. Visualization is saved to `demo_video_results/`, where vertical bars with the same color indicate the same shot.




## <a name="model_zoo"></a> Model Zoo 🤗

| Model | Description | Huggingface |
|-------|-------------|-------------|
| OmniShotCut v1.5 (recommended) | Hidden dim 576, 52.8M params. Best real-world quality: fewer false cuts on continuous camera motion / lighting changes and better dense fast-cut recall. | [Download](https://huggingface.co/uva-cv-lab/OmniShotCut_v1.5) |
| OmniShotCut (paper) | 3 encoders, 34.5M params. Reproduces the paper (Table 1) results. | [Download](https://huggingface.co/uva-cv-lab/OmniShotCut_paper) |
| OmniShotCut (preview) | 6 encoders, 41M params. Early preview release. | [Download](https://huggingface.co/uva-cv-lab/OmniShotCut) |

**v1.5** is a drop-in replacement (same filename and interface): it fires far fewer false cuts on continuous camera motion and lighting/strobe changes, recovers more dense fast cuts, and keeps the same accuracy on clean content.








## Awesome work using OmniShotCut

- [DreamX-Creator](https://arxiv.org/abs/2608.31106): Applies OmniShotCut for fine-grained shot-boundary detection and transition labeling during finetuning-data curation, discarding candidate segments that contain or span a detected transition.
- [Sekai2](https://arxiv.org/abs/2608.09449): Uses OmniShotCut to detect hard cuts and gradual transitions, then partitions continuous perspective shots into clean clips (up to 120s) for world-model pretraining data.
- [ContextMaster](https://arxiv.org/abs/2608.04956): Invokes OmniShotCut to segment multi-shot videos and localize individual shots, both for its Interactive Director tool and for preparing addition/removal/replacement training data.
- [AlayaWorld](https://arxiv.org/abs/2607.18367): Combines classical cut/dissolve detection with OmniShotCut in its data-curation pipeline's shot-boundary filtering stage, ensuring every retained training sample is a single continuous shot.




## 📚 Citation
```bibtex
@article{wang2026omnishotcut,
  title={OmniShotCut: Holistic Relational Shot Boundary Detection with Shot-Query Transformer},
  author={Wang, Boyang and Xu, Guangyi and Zhang, Jiahui and Tang, Zhipeng and Cheng, Zezhou},
  journal={arXiv preprint arXiv:2604.24762},
  year={2026}
}
```


