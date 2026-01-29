import os
import time
import datetime
import numpy as np
import tensorflow as tf
import cv2 as cv
import matplotlib.pyplot as plt
import pandas as pd
import argparse

from model.nir_egcc import NIR_EGCC as MY_MODEL
from dataset_manager import DataLoader
from utils import image_normalization, save_visual_results
from losses import FineTuneLoss


@tf.function
def train_step_fast(model, x, y, optimizer, loss_obj):
    """ Accelerated training step using tf.function """
    with tf.GradientTape() as tape:
        p = model(x, training=True)
        total_loss, l_c, l_s, l_p, l_col = loss_obj.call(y, p)
        scaled_loss = optimizer.get_scaled_loss(total_loss)

    scaled_grads = tape.gradient(scaled_loss, model.trainable_variables)
    grads = optimizer.get_unscaled_gradients(scaled_grads)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return total_loss, l_c, l_s, l_p, l_col, p


def train_finetune(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    segment = "Stage_2"

    # Workspace setup
    time_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    res_dir = os.path.join('results', segment, time_str)
    ckpnt_dir = os.path.join('checkpoints', segment, time_str)
    for d in [res_dir, ckpnt_dir]:
        os.makedirs(d, exist_ok=True)

    tf.keras.mixed_precision.set_global_policy('mixed_float16')

    # Data and model initialization
    data = DataLoader(args.train_list, args.img_height, args.img_width, args.batch_size, model_state='train')

    model = MY_MODEL()
    _ = model(tf.zeros((1, args.img_height, args.img_width, 4)))

    # Load pretrained weights
    if not os.path.exists(args.pretrained_path):
        raise FileNotFoundError(f"Pretrained weights not found: {args.pretrained_path}")
    model.load_weights(args.pretrained_path)
    print(f"Weights loaded from {args.pretrained_path}. Starting finetuning...")

    # Optimizer with low learning rate
    optimizer = tf.keras.mixed_precision.LossScaleOptimizer(
        tf.keras.optimizers.Adam(learning_rate=tf.keras.optimizers.schedules.ExponentialDecay(
            args.lr, 2000, 0.95, staircase=True))
    )

    loss_fn = FineTuneLoss(args.img_height, args.img_width)
    history = []

    print(f"Finetuning started. Target: Color optimization. Results: {res_dir}")

    for epoch in range(args.epochs):
        epoch_loss_avg, prog_col = 0.0, 0.0
        steps = len(data)
        start_time = time.time()

        for step, (x, y) in enumerate(data):
            total, l_c, l_s, l_p, l_col, p = train_step_fast(model, x, y, optimizer, loss_fn)

            loss_val = float(total.numpy())
            epoch_loss_avg += loss_val
            prog_col += float(l_col)

            if step % 10 == 0:
                print(f"\rFT Epoch {epoch} [{step}/{steps}] Total: {loss_val:.4f} | Color: {float(l_col):.4f}", end="")

        avg_loss = epoch_loss_avg / steps
        history.append(avg_loss)
        print(f"\nEpoch {epoch} finished | Avg Loss: {avg_loss:.4f} | Color Loss: {prog_col / steps:.4f} | Time: {time.time() - start_time:.1f}s")

        # Periodic visualization
        if epoch % 2 == 0:
            save_visual_results(x, y, p, epoch, avg_loss, res_dir, image_normalization)

    # Save final finetuned weights
    final_path = os.path.join(ckpnt_dir, "finetuned_final.h5")
    model.save_weights(final_path, save_format='h5')
    print(f"Finetuning complete. Weights saved to: {final_path}")

    # Log metrics
    pd.DataFrame({'Epoch': range(args.epochs), 'Loss': history}).to_csv(os.path.join(res_dir, 'ft_history.csv'), index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NIR_EGCC Fine-tuning")
    parser.add_argument('--pretrained_path', type=str,
                        default="checkpoints/pretrained/my_stage_1.h5",
                        help="Path to pretrained weights")
    parser.add_argument('--train_list', type=str, default="data/train_list.txt")
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--img_width', type=int, default=192)
    parser.add_argument('--img_height', type=int, default=192)
    parser.add_argument('--gpu_id', type=str, default='0')

    args = parser.parse_args()
    try:
        train_finetune(args)
    except Exception as e:
        print(f"\nError: {e}")