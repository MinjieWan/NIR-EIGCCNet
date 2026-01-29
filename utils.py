import os
import cv2
import numpy as np
from skimage.metrics import structural_similarity as compare_ssim

def _safe_ssim(y01: np.ndarray, p01: np.ndarray) -> float:
    """ Version compatibility wrapper for skimage SSIM calculation """
    try:
        return compare_ssim(y01, p01, data_range=1.0, channel_axis=-1)
    except TypeError:
        return compare_ssim(y01, p01, data_range=1.0, multichannel=True)

def _resize_to_height(img: np.ndarray, H: int) -> np.ndarray:
    """ Resize image to target height while maintaining aspect ratio """
    h, w = img.shape[:2]
    if h == 0:
        return img
    new_w = max(1, int(round(w * (H / float(h)))))
    return cv2.resize(img, (new_w, H), interpolation=cv2.INTER_AREA)

def _draw_label(img: np.ndarray, text: str, font_scale=0.7, thickness=2) -> np.ndarray:
    """ Overlay semi-transparent label bar and text onto the image """
    out = img.copy()
    h, w = out.shape[:2]
    bar_h = 26
    overlay = out.copy()
    cv2.rectangle(overlay, (0, 0), (w, bar_h), (30, 30, 30), -1)
    cv2.addWeighted(overlay, 0.6, out, 0.4, 0, out)
    cv2.putText(out, text, (8, bar_h - 8), cv2.FONT_HERSHEY_SIMPLEX,
                font_scale, (0, 255, 0), thickness, cv2.LINE_AA)
    return out

def _hstack_with_gap(images: list, gap: int, border: int) -> np.ndarray:
    """ Concatenate images horizontally with specified gap and border spacing """
    heights = [im.shape[0] for im in images]
    if len(set(heights)) != 1:
        raise ValueError("All images must have identical height for horizontal concatenation")

    H = heights[0]
    gap_strip = np.zeros((H, gap, 3), dtype=np.uint8)

    canvas = images[0]
    for im in images[1:]:
        canvas = np.concatenate([canvas, gap_strip, im], axis=1)

    canvas = cv2.copyMakeBorder(canvas, border, border, border, border,
                                cv2.BORDER_CONSTANT, value=(0, 0, 0))
    return canvas

def _save_pred_files(base_dir: str, stem: str,
                     x_vis: np.ndarray, x_nir: np.ndarray, y_vis: np.ndarray, p_vis: np.ndarray,
                     p_raw_01: np.ndarray,
                     save_individ: bool = True,
                     save_pred_h5: bool = False):
    """ Save prediction outputs to disk in PNG and HDF5 formats """
    x_bgr = cv2.cvtColor(x_vis.astype(np.uint8), cv2.COLOR_RGB2BGR)
    y_bgr = cv2.cvtColor(y_vis.astype(np.uint8), cv2.COLOR_RGB2BGR)
    p_bgr = cv2.cvtColor(p_vis.astype(np.uint8), cv2.COLOR_RGB2BGR)

    if save_individ:
        folders = ['input_rgb_png', 'input_nir_png', 'gt_png', 'pred_png']
        for f in folders:
            os.makedirs(os.path.join(base_dir, f), exist_ok=True)

        cv2.imwrite(os.path.join(base_dir, 'input_rgb_png', f'{stem}.png'), x_bgr)
        cv2.imwrite(os.path.join(base_dir, 'input_nir_png', f'{stem}.png'), x_nir)
        cv2.imwrite(os.path.join(base_dir, 'gt_png', f'{stem}.png'), y_bgr)
        cv2.imwrite(os.path.join(base_dir, 'pred_png', f'{stem}.png'), p_bgr)

    if save_pred_h5:
        import h5py
        os.makedirs(os.path.join(base_dir, 'pred_h5'), exist_ok=True)
        with h5py.File(os.path.join(base_dir, 'pred_h5', f'{stem}.h5'), 'w') as f:
            f.create_dataset('data', data=p_raw_01.astype(np.float32))

def image_normalization(img, img_min=0, img_max=255):
    """
    Linearly normalize image pixel values to a specified target range.
    Uses the following formula:
    $$I_{norm} = \frac{(I - I_{min}) \cdot (Range_{max} - Range_{min})}{(I_{max} - I_{min}) + \epsilon} + Range_{min}$$
    """
    img = np.float32(img)
    epsilon = 1e-12
    img = (img - np.min(img)) * (img_max - img_min) / ((np.max(img) - np.min(img)) + epsilon) + img_min
    return img



def save_visual_results(x, y, p, epoch, avg_loss, res_dir, normalization_fn):
    """ Generate and save a side-by-side visualization of Input, GT, and Prediction """
    bx = 0
    tmp_x = cv2.cvtColor(normalization_fn(np.squeeze(x[bx, :, :, :3])), cv2.COLOR_RGB2BGR)
    tmp_y = cv2.cvtColor(normalization_fn(np.squeeze(y[bx, ...])), cv2.COLOR_RGB2BGR)
    tmp_p = cv2.cvtColor(normalization_fn(p[bx, ...].numpy()), cv2.COLOR_RGB2BGR)

    vis_imgs = np.concatenate((tmp_x, tmp_y, tmp_p), axis=1)
    vis_imgs = np.uint8(vis_imgs)

    cv2.putText(vis_imgs, f"Ep:{epoch} L:{avg_loss:.4f}", (10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    save_path = os.path.join(res_dir, f'viz_epoch_{epoch}.png')
    cv2.imwrite(save_path, vis_imgs)