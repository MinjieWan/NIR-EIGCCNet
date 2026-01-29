import cv2
import numpy as np
import random
import tensorflow as tf
import h5py
import os


def add_gaussian_blur_to_rgb(img_rgb, kernel_size=(5, 5), sigma=1.0):
    """ Apply Gaussian blur to the RGB channels only """
    return cv2.GaussianBlur(img_rgb, kernel_size, sigma)


class DataLoader(tf.keras.utils.Sequence):
    """ Custom DataLoader for NIR-RGB image pairs using HDF5 files """
    def __init__(self,
                 list_path: str,
                 img_height: int,
                 img_width: int,
                 batch_size: int,
                 model_state: str = "train",
                 gamma: float = 0.4545):
        super().__init__()
        self.list_path = os.path.abspath(list_path)
        self.base_dir = os.path.dirname(self.list_path)
        self.dim_h = int(img_height)
        self.dim_w = int(img_width)
        self.bs = int(batch_size)
        self.model_state = str(model_state).lower()
        self.is_training = (self.model_state == "train")
        self.gamma = float(gamma)

        self.data_list = self._build_index()
        in_list, _ = self.data_list
        assert len(in_list) > 0, f"Empty or invalid list file: {self.list_path}"

        if not self.is_training:
            self.imgs_name = [os.path.basename(p) for p in in_list]
            self.input_shape = (None, self.dim_h, self.dim_w, 4)

        self.indices = np.arange(len(in_list))
        if self.is_training:
            np.random.shuffle(self.indices)

    def _build_index(self):
        """ Parse the list file to resolve absolute paths for input and GT pairs """
        input_paths, gt_paths = [], []
        if not os.path.isfile(self.list_path):
            raise FileNotFoundError(f"List file not found: {self.list_path}")

        with open(self.list_path, 'r', encoding='utf-8') as f:
            for lineno, line in enumerate(f, 1):
                s = line.strip()
                if not s:
                    continue
                parts = s.split()
                if len(parts) != 2:
                    raise ValueError(f"[Line {lineno}] Format error: {s!r}")

                in_p, gt_p = parts
                in_abs = in_p if os.path.isabs(in_p) else os.path.normpath(os.path.join(self.base_dir, in_p))
                gt_abs = gt_p if os.path.isabs(gt_p) else os.path.normpath(os.path.join(self.base_dir, gt_p))
                in_abs = os.path.abspath(in_abs)
                gt_abs = os.path.abspath(gt_abs)

                if not os.path.isfile(in_abs) or not os.path.isfile(gt_abs):
                    raise FileNotFoundError(f"File missing at line {lineno}: {in_abs} or {gt_abs}")

                input_paths.append(in_abs)
                gt_paths.append(gt_abs)

        return [input_paths, gt_paths]

    def on_epoch_end(self):
        if self.is_training:
            np.random.shuffle(self.indices)

    def __len__(self):
        return len(self.indices) // self.bs

    def __getitem__(self, index):
        idx = self.indices[index * self.bs: (index + 1) * self.bs]
        x_list, y_list = self.data_list
        tmp_x_path = [x_list[k] for k in idx]
        tmp_y_path = [y_list[k] for k in idx]
        x, y = self.__data_generation(tmp_x_path, tmp_y_path)
        return x, y

    def __data_generation(self, x_path_list, y_path_list):
        """ Read, crop, and preprocess a batch of samples """
        x = np.empty((self.bs, self.dim_h, self.dim_w, 4), dtype=np.float32)
        y = np.empty((self.bs, self.dim_h, self.dim_w, 3), dtype=np.float32)

        for i, (xp, yp) in enumerate(zip(x_path_list, y_path_list)):
            xi, yi = self.transformer(xp, yp)

            # Apply gamma correction
            xi = tf.pow(tf.convert_to_tensor(xi, dtype=tf.float32), self.gamma).numpy()
            yi = tf.pow(tf.convert_to_tensor(yi, dtype=tf.float32), self.gamma).numpy()

            # Process channels
            tmp_x_rgb = xi[:, :, :3]
            tmp_x_nir = xi[:, :, 3:]

            # Apply blur to RGB only
            tmp_x_rgb = add_gaussian_blur_to_rgb(tmp_x_rgb)
            xi = np.concatenate([tmp_x_rgb, tmp_x_nir], axis=-1)

            x[i] = xi
            y[i] = yi

        return x, y

    def transformer(self, x_path, y_path):
        """ Load H5 data and apply cropping (Random for train, Center for test) """
        tmp_x = self.__read_h5(x_path)
        tmp_y = self.__read_h5(y_path)

        if tmp_x.ndim != 3 or tmp_y.ndim != 3:
            raise ValueError(f"Invalid dimensions: {x_path}")

        h, w = tmp_x.shape[:2]
        if h < self.dim_h or w < self.dim_w:
            raise ValueError(f"Image too small: {(h, w)} < {(self.dim_h, self.dim_w)}")

        if self.is_training:
            top = random.randint(0, h - self.dim_h)
            left = random.randint(0, w - self.dim_w)
        else:
            top = (h - self.dim_h) // 2
            left = (w - self.dim_w) // 2

        tmp_x = tmp_x[top: top + self.dim_h, left: left + self.dim_w, :]
        tmp_y = tmp_y[top: top + self.dim_h, left: left + self.dim_w, :]

        return tmp_x, tmp_y

    @staticmethod
    def __read_h5(file_path: str) -> np.ndarray:
        """ Read array data from HDF5 file """
        with h5py.File(file_path, 'r') as h5f:
            if 'data' in h5f:
                arr = h5f['data'][:]
            else:
                first_key = next(iter(h5f.keys()))
                arr = h5f[first_key][:]
        return np.asarray(arr, dtype=np.float32)