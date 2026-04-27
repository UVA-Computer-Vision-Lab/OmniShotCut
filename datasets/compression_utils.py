import random
import io
import cv2
import numpy as np
from PIL import Image
import torch
cv2.setNumThreads(0)



# Setting
jpeg_quality_range = (70, 90)               # Higher, Better Quality
webp_quality_range = (70, 90)
webp_encode_speed  = (3, 5)                 # Higher, Slower, Better Quality



def _to_hwc_uint8(x: torch.Tensor) -> np.ndarray:
    t = x.detach()

    keep_batch_dim = False
    if t.ndim == 4:
        keep_batch_dim = True
        if t.shape[0] != 1:
            raise ValueError(f"Expect batch size 1 if 4D, got {tuple(t.shape)}")
        t = t.squeeze(0)

    if t.ndim != 3:
        raise ValueError(f"Expect 3D tensor after squeeze, got {tuple(t.shape)}")

    if t.shape[0] == 3:
        t = t.permute(1, 2, 0)  # CHW -> HWC
    elif t.shape[2] == 3:
        pass  # already HWC
    else:
        raise ValueError(f"Expect CHW or HWC with 3 channels, got {tuple(t.shape)}")

    if t.dtype != torch.uint8:
        t = (t.clamp(0, 1) * 255.0).round().to(torch.uint8)

    arr = t.cpu().numpy()  # HWC uint8, usually contiguous
    # ensure contiguous anyway
    return np.ascontiguousarray(arr)


def _from_hwc_uint8(img: np.ndarray, keep_batch_dim: bool) -> torch.Tensor:
    if img.ndim != 3 or img.shape[2] != 3:
        raise ValueError(f"Expect (H,W,3), got {img.shape}")
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)

    img = np.ascontiguousarray(img)  # <-- critical for safety
    t = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0  # (3,H,W)
    return t.unsqueeze(0) if keep_batch_dim else t



def jpeg_compress_tensor(tensor_frames: torch.Tensor) -> torch.Tensor:
    keep_batch_dim = (tensor_frames.ndim == 4)
    img_rgb = _to_hwc_uint8(tensor_frames)

    # BGR for OpenCV (make contiguous to avoid negative strides)
    img_bgr = np.ascontiguousarray(img_rgb[..., ::-1])

    q = random.randint(jpeg_quality_range[0], jpeg_quality_range[1])

    ok, enc = cv2.imencode(".jpg", img_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(q)])
    if not ok:
        raise RuntimeError("cv2.imencode('.jpg') failed")

    dec_bgr = cv2.imdecode(enc, cv2.IMREAD_COLOR)
    if dec_bgr is None:
        raise RuntimeError("cv2.imdecode() failed")

    # back to RGB (again: contiguous!)
    dec_rgb = np.ascontiguousarray(dec_bgr[..., ::-1])
    return _from_hwc_uint8(dec_rgb, keep_batch_dim)



def webp_compress_tensor(tensor_frames: torch.Tensor) -> torch.Tensor:
    keep_batch_dim = (tensor_frames.ndim == 4)
    img_rgb = _to_hwc_uint8(tensor_frames)

    quality = random.randint(webp_quality_range[0], webp_quality_range[1])
    method  = random.randint(webp_encode_speed[0], webp_encode_speed[1])

    im = Image.fromarray(img_rgb, mode="RGB")
    buf = io.BytesIO()
    im.save(buf, format="WEBP", quality=int(quality), method=int(method))
    data = buf.getvalue()

    dec = np.array(Image.open(io.BytesIO(data)).convert("RGB"), dtype=np.uint8)
    dec = np.ascontiguousarray(dec)  # safety
    return _from_hwc_uint8(dec, keep_batch_dim)
