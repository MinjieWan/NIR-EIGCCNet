import os
import time
import datetime
import tensorflow as tf
import tensorflow.keras as tfk
import matplotlib.pyplot as plt
import pandas as pd
import argparse

from model.nir_egcc import NIR_EGCC as MY_MODEL
from dataset_manager import DataLoader
from utils import image_normalization, save_visual_results
from losses import AdvancedLoss


@tf.function
def train_step_fast(model, x, y, optimizer, loss_obj):
    """ Accelerated training step using tf.function """
    with tf.GradientTape() as tape:
        p = model(x, training=True)
        total_loss, l_char, l_ssim, l_per, l_col = loss_obj.call(y, p)
        scaled_loss = optimizer.get_scaled_loss(total_loss)

    scaled_grads = tape.gradient(scaled_loss, model.trainable_variables)
    grads = optimizer.get_unscaled_gradients(scaled_grads)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return total_loss, l_char, l_ssim, l_per, l_col, p


def train(args):
    # Setup environment
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    segment = "Stage_1"

    # Prepare directories
    time_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    res_dir = os.path.join('results', segment, time_str)
    ckpnt_dir = os.path.join('checkpoints', segment, time_str)
    for d in [res_dir, ckpnt_dir]:
        os.makedirs(d, exist_ok=True)

    # Mixed precision policy
    tf.keras.mixed_precision.set_global_policy('mixed_float16')

    # Data and model initialization
    print("Loading data...")
    data = DataLoader(
        list_path=args.train_list,
        img_height=args.img_height,
        img_width=args.img_width,
        batch_size=args.batch_size,
        model_state='train',
    )
    model = MY_MODEL()

    # Optimizer setup
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=args.lr,
        decay_steps=10000,
        decay_rate=0.92,
        staircase=True
    )
    optimizer = tf.keras.mixed_precision.LossScaleOptimizer(
        tfk.optimizers.Adam(learning_rate=lr_schedule, beta_1=0.9, beta_2=0.999)
    )

    advanced_loss = AdvancedLoss(args.img_height, args.img_width)
    train_loss_history = []

    print(f"Training started. Results will be saved to: {res_dir}")

    for epoch in range(args.epochs):
        epoch_loss_avg, prog_char, prog_ssim, prog_per, prog_col = 0.0, 0.0, 0.0, 0.0, 0.0
        steps = len(data)
        start_time = time.time()

        for step, (x, y) in enumerate(data):
            total, l_c, l_s, l_p, l_col, p = train_step_fast(model, x, y, optimizer, advanced_loss)

            # Metric accumulation
            loss_val = float(total.numpy())
            epoch_loss_avg += loss_val
            prog_char += float(l_c)
            prog_ssim += float(l_s)
            prog_per += float(l_p)
            prog_col += float(l_col)

            if step % 10 == 0:
                print(f"\rEpoch {epoch} [{step}/{steps}] Loss: {loss_val:.4f}", end="")

        avg_loss = epoch_loss_avg / steps
        train_loss_history.append(avg_loss)

        print(f"\nEpoch {epoch} finished | Avg Loss: {avg_loss:.4f} | Time: {time.time() - start_time:.1f}s")
        print(
            f"   Breakdown -> Char: {prog_char / steps:.4f}, SSIM: {prog_ssim / steps:.4f}, Per: {prog_per / steps:.1f}, Col: {prog_col / steps:.4f}")

        # Periodic visualization
        if epoch % 2 == 0:
            save_visual_results(x, y, p, epoch, avg_loss, res_dir, image_normalization)

    # Save final model weights
    final_model_path = os.path.join(ckpnt_dir, "final_weights.h5")
    model.save_weights(final_model_path, save_format='h5')
    print(f"Training complete. Weights saved to: {final_model_path}")

    # Log history
    pd.DataFrame({'Epoch': range(args.epochs), 'Total_Loss': train_loss_history}).to_csv(
        os.path.join(res_dir, 'loss_history.csv'), index=False)

    plt.figure(figsize=(10, 6))
    plt.plot(train_loss_history)
    plt.title('Training Process')
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(res_dir, 'loss_curve.png'))
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NIR_EGCC Training Pipeline")
    parser.add_argument('--train_list', type=str, default="data/train_list.txt")
    parser.add_argument('--img_width', type=int, default=192)
    parser.add_argument('--img_height', type=int, default=192)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--gpu_id', type=str, default='0')

    args = parser.parse_args()

    try:
        train(args)
    except KeyboardInterrupt:
        print("\nTraining stopped by user.")
    except Exception as e:
        print(f"\nError occurred: {e}")