import os, sys, math
from typing import List, Optional, Tuple, Dict
import numpy as np
from PIL import Image, ImageDraw, ImageFont




PALETTE: List[Tuple[int, int, int]] = [
    (255, 80, 80),
    (80, 255, 80),
    (80, 160, 255),
    (255, 200, 80),
    (200, 80, 255),
    (80, 255, 220),
]





def _frame_idx_to_mmss(frame_idx: int, fps: float) -> str:
    sec = int(frame_idx / fps)
    mm = sec // 60
    ss = sec % 60
    return f"{mm:02d}:{ss:02d}"




# -------------------------
# Visualization utilities
# -------------------------
def _to_uint8_rgb(frame: np.ndarray) -> np.ndarray:
    if frame.dtype != np.uint8:
        frame = np.clip(frame * 255.0, 0, 255).astype(np.uint8)

    if frame.ndim == 2:
        frame = np.stack([frame] * 3, axis=-1)
    elif frame.ndim == 3 and frame.shape[-1] == 4:
        frame = frame[..., :3]
    return frame


def _load_font(font_size: int):
    try:
        return ImageFont.truetype("DejaVuSans.ttf", font_size)
    except Exception:
        return ImageFont.load_default()


def _build_closed_boundary_color_maps(
    ranges_closed: List[Tuple[int, int]],
    T: int,
    palette: List[Tuple[int, int, int]],
    end_exclusive = False,
):
    start_map, end_map = {}, {}
    for k, (s, e) in enumerate(ranges_closed):

        if end_exclusive:
            e = e - 1

        if s > e:
            s, e = e, s
        if e < 0 or s >= T:
            continue
        s = max(0, s)
        e = min(T - 1, e)
        rgba = (*palette[k % len(palette)], 255)
        start_map[s] = rgba
        end_map[e] = rgba
    return start_map, end_map




def visualize_concated_frames(
    frames: np.ndarray,
    out_dir: str,
    highlight_ranges_closed: Optional[List[Tuple[int, int]]],
    max_frames_per_img: int = 600,
    cols: int = 12,
    pad: int = 3,
    bg_color: Tuple[int, int, int] = (0, 0, 0),
    resize_to: Optional[Tuple[int, int]] = None,
    start_index: int = 0,           # time 的对齐的位置
    text_color: Tuple[int, int, int] = (255, 0, 0),
    font_size: int = 18,
    text_pad: Tuple[int, int] = (4, 2),
    draw_text_bg: bool = True,
    text_bg_rgba: Tuple[int, int, int, int] = (0, 0, 0, 160),
    out_prefix: str = "concat_",
    out_ext: str = ".jpg",
    jpg_quality: int = 75,
    bar_thickness: int = 14,
    palette: Optional[List[Tuple[int, int, int]]] = PALETTE,
    fps: Optional[float] = 24,
    end_range_exclusive = False,             # Whether end range is inclusive or exclusive
    verbose = False,
):

    os.makedirs(out_dir, exist_ok=True)
    font = _load_font(font_size)

    f0 = _to_uint8_rgb(frames[0])
    tile_h, tile_w = f0.shape[:2]
    frames_per_page = max_frames_per_img
    rows = math.ceil(frames_per_page / cols)

    canvas_w = cols * tile_w + (cols - 1) * pad
    canvas_h = rows * tile_h + (rows - 1) * pad

    def new_canvas():
        im = Image.new("RGB", (canvas_w, canvas_h), color=bg_color)
        return im, ImageDraw.Draw(im, "RGBA")

    start_map, end_map = ({}, {})
    if highlight_ranges_closed:
        start_map, end_map = _build_closed_boundary_color_maps(highlight_ranges_closed, len(frames), palette, end_exclusive=end_range_exclusive)

    page = 0
    global_idx = start_index
    canvas, draw = new_canvas()
    saved_paths = []

    # Iterate
    for i, fr in enumerate(frames):
        local_i = i % frames_per_page

        # Store
        if local_i == 0 and i > 0:
            saved_path = os.path.join(out_dir, f"{out_prefix}{page:04d}{out_ext}")
            canvas.save(saved_path, quality=jpg_quality)
            saved_paths.append(saved_path)

            # Update
            page += 1
            canvas, draw = new_canvas()
            

        r = local_i // cols
        c = local_i % cols
        x = c * (tile_w + pad)
        y = r * (tile_h + pad)

        pil = Image.fromarray(_to_uint8_rgb(fr))
        canvas.paste(pil, (x, y))

        if i in start_map:
            draw.rectangle((x, y, x + bar_thickness, y + tile_h), fill=start_map[i])
        if i in end_map:
            draw.rectangle((x + tile_w - bar_thickness, y, x + tile_w, y + tile_h), fill=end_map[i])

        text1 = str(global_idx)
        text2 = _frame_idx_to_mmss(global_idx, fps) if fps is not None else ""

        tx, ty = x + text_pad[0], y + text_pad[1]
        if draw_text_bg:
            bbox = draw.textbbox((tx, ty), text1 + "\n" + text2, font=font)
            draw.rectangle(bbox, fill=text_bg_rgba)

        draw.text((tx, ty), f"{text1}\n{text2}", font=font, fill=text_color)

        global_idx += 1


    # Save the last one
    saved_path = os.path.join(out_dir, f"{out_prefix}{page:04d}{out_ext}")
    canvas.save(saved_path, quality=jpg_quality)
    saved_paths.append(saved_path)

    
    # if verbose:
    #     print(f"Done. Total frames: {len(frames)} | Pages: {page + 1}")


    return saved_paths





def concat_image_lists_horizontal(
    list1: List[str],
    list2: List[str],
    out_dir: str,
    bar_width: int = 40,
    bar_color: Tuple[int, int, int] = (255, 0, 0),
    out_prefix: str = "merged_",
    out_ext: str = ".jpg",
    jpg_quality: int = 90,
    resize_mode: str = "match_height",  # ["match_height", "match_width", "none"]
    verbose: bool = True,
) -> List[str]:
    """
    Horizontally concatenate images from two path lists (same index),
    with a thick visual bar in between.

    Args:
        list1, list2: list of image file paths (must have same length)
        out_dir: directory to save merged images
        bar_width: thickness of the separator bar
        bar_color: RGB color of separator bar
        resize_mode:
            - "match_height": resize second image to match height
            - "match_width": resize second image to match width
            - "none": no resize (heights must match)
    Returns:
        List of saved image paths
    """

    assert len(list1) == len(list2), "Two lists must have same length"

    os.makedirs(out_dir, exist_ok=True)

    saved_paths = []

    for idx, (p1, p2) in enumerate(zip(list1, list2)):

        img1 = Image.open(p1).convert("RGB")
        img2 = Image.open(p2).convert("RGB")

        # ---------- Resize logic ----------
        if resize_mode == "match_height":
            if img1.height != img2.height:
                new_w = int(img2.width * img1.height / img2.height)
                img2 = img2.resize((new_w, img1.height), Image.BILINEAR)

        elif resize_mode == "match_width":
            if img1.width != img2.width:
                new_h = int(img2.height * img1.width / img2.width)
                img2 = img2.resize((img1.width, new_h), Image.BILINEAR)

        elif resize_mode == "none":
            assert img1.height == img2.height, \
                "Heights must match when resize_mode='none'"

        else:
            raise ValueError("resize_mode must be one of ['match_height', 'match_width', 'none']")

        # ---------- Create bar ----------
        bar = Image.new("RGB", (bar_width, img1.height), color=bar_color)

        # ---------- Create canvas ----------
        total_width = img1.width + bar_width + img2.width
        canvas = Image.new("RGB", (total_width, img1.height))

        canvas.paste(img1, (0, 0))
        canvas.paste(bar, (img1.width, 0))
        canvas.paste(img2, (img1.width + bar_width, 0))

        # ---------- Save ----------
        out_path = os.path.join(out_dir, f"{out_prefix}{idx:04d}{out_ext}")
        canvas.save(out_path, quality=jpg_quality)
        saved_paths.append(out_path)

    if verbose:
        print(f"Done. Saved {len(saved_paths)} merged images to {out_dir}")


    return saved_paths


