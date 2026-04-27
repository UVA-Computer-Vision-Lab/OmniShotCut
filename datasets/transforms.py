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
import torchvision.transforms.functional as F
from torchvision.transforms import InterpolationMode
from torch.utils.data import Dataset
from PIL import Image
PILImage = Image.Image




def _to_4d_video_tensor(frames_np):
    """
    Inputs: [N, H, W, 3]  in range [0, 255]
    Returns: float tensor [N, C, H, W] in [0, 1]
    """

    # numpy -> torch, NHWC -> NCHW
    frames = torch.from_numpy(frames_np.copy())

    # Divide 255 to the range [0, 1]
    frames = frames.float().div_(255.0)

    # Reshape to [T,C,H,W]
    frames = frames.permute(0, 3, 1, 2).contiguous()        # [T,C,H,W]

    # Return
    return frames





def save_video_mp4(
    video: torch.Tensor,
    path: str = "video.mp4",
    fps: int = 24,
    mean=(0.485, 0.456, 0.406),
    std=(0.229, 0.224, 0.225),
    assume_normalized: bool | None = None,  # None=自动判断；True/False=强制
):
    """
    Save torch video tensor to mp4.

    Accepts:
      - [T, C, H, W] float tensor
      - [B, T, C, H, W] float tensor (will use first sample)
    It will auto unnormalize (ImageNet) if it detects normalized inputs.

    Output:
      - mp4 with uint8 frames [T, H, W, 3]
    """

    if not torch.is_tensor(video):
        raise TypeError(f"video must be torch.Tensor, got {type(video)}")

    v = video.detach().cpu()

    # Handle [B, T, C, H, W]
    if v.ndim == 5:
        v = v[0]
    if v.ndim != 4:
        raise ValueError(f"Expected [T,C,H,W] (or [B,T,C,H,W]), got shape {tuple(v.shape)}")

    T, C, H, W = v.shape
    if C not in (1, 3):
        raise ValueError(f"Expected C=1 or 3, got C={C}")

    v = v.to(torch.float32)

    # ---- Decide whether to unnormalize ----
    # Heuristic: normalized ImageNet tensors often have values outside [0,1]
    # (e.g., negative or >1). Raw unnormalized typically stays in [0,1].
    if assume_normalized is None:
        minv = float(v.min())
        maxv = float(v.max())
        is_normalized = (minv < -0.05) or (maxv > 1.05)
    else:
        is_normalized = bool(assume_normalized)

    if is_normalized:
        mean_t = torch.tensor(mean, dtype=v.dtype).view(1, 3, 1, 1)
        std_t  = torch.tensor(std,  dtype=v.dtype).view(1, 3, 1, 1)
        if C == 1:
            v = v.repeat(1, 3, 1, 1)
            C = 3
        v = v * std_t + mean_t  # unnormalize back to roughly [0,1]

    # Clamp to valid range and convert to uint8
    v = v.clamp(0.0, 1.0)
    v = (v * 255.0).round().to(torch.uint8)

    # [T, C, H, W] -> [T, H, W, C]
    v = v.permute(0, 2, 3, 1).contiguous().numpy()

    imageio.mimsave(path, v, fps=fps)
    # print("Save video at", path)





class Video_Augmentation_Transform:
    """
        Clip-wise (video-constant) augmentation.
        All randomness sampled ONCE per call, applied identically to every frame.
    """

    def __init__(
        self,
        set_type: str = "train",
        horizontal_flip_prob: float = 0.5,
        vertical_flip_prob: float = 0.1,
        jitter_prob: float = 0.5,
        jitter_param: Tuple[float, float, float, float] = (0.2, 0.2, 0.2, 0.05),  # b,c,s,h
        grayscale_prob: float = 0.1,
        # ---- add blur ----
        blur_prob: float = 0.1,
        blur_kernel_size: int = 8,
        blur_sigma: Tuple[float, float] = (0.1, 2.0),
        # ---- add noise ----
        noise_prob: float = 0.2,
        noise_sigma: Tuple[float, float] = (0.003, 0.01),  # for [0,1] images
        noise_clip: Tuple[float, float] = (0.0, 1.0),
        # ---- add compression ----
        compression_prob = 0.0,
        compression_choices = ["jpeg", "webp"],
        # ---- Basic Normlization ----
        normalize_mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
        normalize_std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
    ):
        
        self.set_type = set_type
        self.horizontal_flip_prob = horizontal_flip_prob
        self.vertical_flip_prob = vertical_flip_prob
        self.jitter_prob = jitter_prob
        self.jitter_param = jitter_param
        self.grayscale_prob = grayscale_prob
        self.blur_prob = blur_prob
        self.blur_kernel_size = blur_kernel_size
        self.blur_sigma = blur_sigma
        self.noise_prob = noise_prob
        self.noise_sigma = noise_sigma
        self.noise_clip = noise_clip
        self.compression_prob = compression_prob
        self.compression_choices = compression_choices
        self.mean = normalize_mean
        self.std = normalize_std


    @staticmethod
    def _sample_color_jitter_params(b, c, s, h):

        # Sample factors similar to torchvision ColorJitter
        def _rand_factor(x):
            if x is None or x == 0:
                return None
            lo = max(0.0, 1.0 - x)
            hi = 1.0 + x
            return random.uniform(lo, hi)

        brightness_factor = _rand_factor(b)
        contrast_factor = _rand_factor(c)
        saturation_factor = _rand_factor(s)
        hue_factor = None
        if h is not None and h != 0:
            hue_factor = random.uniform(-h, h)

        # Randomize application order (same as torchvision)
        order = ["brightness", "contrast", "saturation", "hue"]
        random.shuffle(order)

        return {
                    "brightness": brightness_factor,
                    "contrast": contrast_factor,
                    "saturation": saturation_factor,
                    "hue": hue_factor,
                    "order": order,
                }


    @staticmethod
    def _fix_blur_kernel(k: int) -> int:

        # torchvision GaussianBlur typically expects odd kernel
        k = int(k)
        if k <= 0:
            raise ValueError("blur_kernel_size must be > 0")
        if k % 2 == 0:
            k = k + 1

        return k


    @staticmethod
    def _add_gaussian_noise(img: torch.Tensor, sigma: float, clip_min: float = 0.0, clip_max: float = 1.0) -> torch.Tensor:
        """
        img: [C,H,W] float, assumed in [0,1] (or at least bounded)
        sigma: std of Gaussian noise in same scale as img
        """
        if sigma <= 0:
            return img
        noise = torch.randn_like(img) * float(sigma)
        img = img + noise
        return img.clamp_(clip_min, clip_max)


    def _add_compression(self, img, compression_choice):

        if compression_choice == "jpeg":
            from datasets.compression_utils import jpeg_compress_tensor

            # compress
            jpeg_compress_tensor(img)
            compressed_img = jpeg_compress_tensor(img)
            
        elif compression_choice == "webp":
            from datasets.compression_utils import webp_compress_tensor

            # compress
            compressed_img = webp_compress_tensor(img)

        else:
            raise NotImplementedError("We do not support comrpession type of", compression_choice)
        
        return compressed_img



    def __call__(self, frames: Union[List[PILImage], torch.Tensor], idx = 0) -> torch.Tensor:
        """
        Returns normalized float tensor [T, C, H, W]
        """

        # Convert to Tesnor
        x = _to_4d_video_tensor(frames)  # [T, C, H, W] in float
        Tt, C, H, W = x.shape


        # Decide the prob
        if self.set_type == "train":
        
            ## Flip
            do_horizontal_flip = random.random() < self.horizontal_flip_prob
            do_vertical_flip = random.random() < self.vertical_flip_prob

            ## Color Jitter
            do_jitter = random.random() < self.jitter_prob
            b, c, s, h = self.jitter_param
            jitter_params = self._sample_color_jitter_params(b, c, s, h)

            ## GaryScale
            do_gray = random.random() < self.grayscale_prob

            ## Blur
            do_blur = random.random() < self.blur_prob
            blur_kernel = self._fix_blur_kernel(self.blur_kernel_size)
            blur_sigma = random.uniform(self.blur_sigma[0], self.blur_sigma[1])

            ## Gaussian Noise
            do_noise = random.random() < self.noise_prob
            noise_sigma = random.uniform(self.noise_sigma[0], self.noise_sigma[1]) if do_noise else 0.0
            clip_min, clip_max = self.noise_clip

            ## Compression
            do_compression = random.random() < self.compression_prob
            compression_choice = random.choice(self.compression_choices)

        else:       # For Testing, No Augmentation at all

            do_horizontal_flip = False
            do_vertical_flip = False
            do_jitter = False
            do_gray = False
            do_blur = False
            do_noise = False
            do_compression = False



        # Apply per-frame with shared params 
        out = []
        for t in range(Tt):     # Iterate each Frame
            img = x[t]          # torch shape is [C, H, W]

            ## Horizontal Flipping
            if do_horizontal_flip:
                img = F.hflip(img)

            if do_vertical_flip:
                img = F.vflip(img)


            ## Color Jitter (shared factors + shared order)
            if do_jitter:
                for op in jitter_params["order"]:      
                    if op == "brightness" and jitter_params["brightness"] is not None:
                        img = F.adjust_brightness(img, jitter_params["brightness"])
                    elif op == "contrast" and jitter_params["contrast"] is not None:
                        img = F.adjust_contrast(img, jitter_params["contrast"])
                    elif op == "saturation" and jitter_params["saturation"] is not None:
                        img = F.adjust_saturation(img, jitter_params["saturation"])
                    elif op == "hue" and jitter_params["hue"] is not None:
                        img = F.adjust_hue(img, jitter_params["hue"])

            ## Gray Scale
            if do_gray:
                img = F.rgb_to_grayscale(img, num_output_channels=3)

            # Gaussian Blur
            if do_blur:
                img = F.gaussian_blur(img, kernel_size=[blur_kernel, blur_kernel], sigma=[blur_sigma, blur_sigma])

            # Gaussian Noise (small, clip-wise sigma)
            if do_noise:
                img = self._add_gaussian_noise(img, sigma=noise_sigma, clip_min=clip_min, clip_max=clip_max)

            # Image compression
            if do_compression:
                img = self._add_compression(img, compression_choice)
                

            # ImageNet normalize per-frame    (Must Do)
            img = F.normalize(img, mean = self.mean, std = self.std)


            # Append
            out.append(img)

        # Stack 
        out = torch.stack(out, dim=0)       # Output Shape is [T, C, H, W]
        

        # Save the video (Comment Out Later)
        # write_path = "augmented"+str(idx)+".mp4"
        # save_video_mp4(out, write_path)


        # Return
        return out         



