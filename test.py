import os
import numpy as np
import tensorflow as tf
import cv2 as cv
import argparse
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.color import rgb2lab, deltaE_ciede2000

from model.nir_egcc import NIR_EGCC as MY_MODEL
from dataset_manager import DataLoader
from utils import (
    _safe_ssim,
    _resize_to_height,
    _draw_label,
    _hstack_with_gap,
    _save_pred_files,
    image_normalization
)

# Configuration for visualization and saving
RESULTS_ROOT = 'results'
SAVE_INDIVID = True
SAVE_PRED_H5 = False
VIS_HEIGHT = 384
TILE_GAP = 8
BORDER = 8


def run_test(args):
    # Path parsing
    CKPT_PATH = os.path.abspath(args.ckpt_path)
    if not os.path.isfile(CKPT_PATH):
        raise FileNotFoundError(f"Checkpoint not found: {CKPT_PATH}")

    norm_path = os.path.normpath(CKPT_PATH)
    path_parts = norm_path.split(os.sep)
    CKPT_NUMBER = os.path.splitext(path_parts[-1])[0]
    CKPT_TIME = path_parts[-2]
    MODEL_NAME = path_parts[-3]

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

    # Data loader initialization
    data4testing = DataLoader(
        list_path=args.test_list,
        img_height=args.img_height,
        img_width=args.img_width,
        batch_size=args.batch_size,
        model_state='test',
    )

    # Directory setup
    res_dir = os.path.join(RESULTS_ROOT, MODEL_NAME, CKPT_TIME, CKPT_NUMBER)
    os.makedirs(res_dir, exist_ok=True)
    montage_dir = os.path.join(res_dir, "montage")
    os.makedirs(montage_dir, exist_ok=True)

    # Model initialization
    model = MY_MODEL()
    _ = model(tf.zeros((1, args.img_height, args.img_width, 4), dtype=tf.float32), training=False)
    model.load_weights(CKPT_PATH)

    # Evaluation metrics
    imgs_psnr, imgs_ssim, imgs_deltae = [], [], []
    imgs_name = getattr(data4testing, 'imgs_name', None)

    print(f"Starting test. Metrics: PSNR, SSIM, DeltaE00")

    for i, (x, y) in enumerate(data4testing):
        p = model(x, training=False).numpy()
        x0, y0, p0 = x[0], y[0], p[0]

        p01 = image_normalization(p0, img_min=0., img_max=1.)
        y01 = image_normalization(y0, img_min=0., img_max=1.)

        if p01.shape != y01.shape:
            p01 = tf.image.resize(p01, size=y01.shape[:2], method='bilinear').numpy()

        # Core metric calculation
        psnr_val = compare_psnr(y01, p01, data_range=1.0)
        ssim_val = _safe_ssim(y01, p01)

        # DeltaE00 calculation
        y_lab = rgb2lab(y01)
        p_lab = rgb2lab(p01)
        delta_e_mean = float(np.mean(deltaE_ciede2000(y_lab, p_lab)))

        imgs_psnr.append(psnr_val)
        imgs_ssim.append(ssim_val)
        imgs_deltae.append(delta_e_mean)

        name = imgs_name[i] if imgs_name is not None else f"img_{i:05d}.h5"
        stem = os.path.splitext(name)[0]
        print(
            f"[{i}/{len(data4testing)}] {name} | PSNR: {psnr_val:.2f} | SSIM: {ssim_val:.4f} | dE00: {delta_e_mean:.2f}")

        # Visualization and storage
        x_vis = image_normalization(x0[:, :, :3]).astype(np.uint8)
        x_nir = image_normalization(x0[:, :, 3:4]).astype(np.uint8)
        y_vis = image_normalization(y01).astype(np.uint8)
        p_vis = image_normalization(p01).astype(np.uint8)

        _save_pred_files(res_dir, stem, x_vis, x_nir, y_vis, p_vis, p01, SAVE_INDIVID, SAVE_PRED_H5)

        x_show = _draw_label(_resize_to_height(x_vis, VIS_HEIGHT), f"In | {stem}")
        y_show = _draw_label(_resize_to_height(y_vis, VIS_HEIGHT), "GT")
        p_show = _draw_label(_resize_to_height(p_vis, VIS_HEIGHT),
                             f"Pred | P:{psnr_val:.2f} S:{ssim_val:.4f} dE:{delta_e_mean:.2f}")

        montage_rgb = _hstack_with_gap([x_show, y_show, p_show], TILE_GAP, BORDER)
        cv.imwrite(os.path.join(montage_dir, f"vis_{stem}.png"), cv.cvtColor(montage_rgb, cv.COLOR_RGB2BGR))

    # Summary output
    m_psnr, m_ssim, m_de = np.mean(imgs_psnr), np.mean(imgs_ssim), np.mean(imgs_deltae)

    print('\n' + '=' * 40)
    print(f"Final Results:")
    print(f"Mean PSNR: {m_psnr:.4f}")
    print(f"Mean SSIM: {m_ssim:.4f}")
    print(f"Mean dE00: {m_de:.4f}")
    print('=' * 40)

    # Save evaluation results to file
    with open(os.path.join(res_dir, 'evaluation_results.txt'), 'w', encoding='utf-8') as f:
        f.write(f"Model: {MODEL_NAME}\n")
        f.write(f"Images: {len(imgs_psnr)}\n")
        f.write(f"Mean PSNR: {m_psnr:.4f}\n")
        f.write(f"Mean SSIM: {m_ssim:.4f}\n")
        f.write(f"Mean DeltaE00: {m_de:.4f}\n")
        f.write("-" * 30 + "\n")
        f.write("Detailed Results (Name, PSNR, SSIM, DeltaE00):\n")
        for i, name in enumerate(imgs_name):
            f.write(f"{name}: {imgs_psnr[i]:.4f}, {imgs_ssim[i]:.4f}, {imgs_deltae[i]:.4f}\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Inference and Evaluation Script")
    parser.add_argument('--ckpt_path', type=str, default="checkpoints/pretrained/my_stage_1.h5")
    parser.add_argument('--test_list', type=str, default="data/test_list.txt")
    parser.add_argument('--img_width', type=int, default=576)
    parser.add_argument('--img_height', type=int, default=320)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--gpu_id', type=str, default='0')
    args = parser.parse_args()

    try:
        run_test(args)
    except KeyboardInterrupt:
        print("\nTest interrupted by user.")
    except Exception as e:
        print(f"\nError: {e}")