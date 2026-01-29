import tensorflow as tf
from tensorflow.keras.applications import VGG19
from tensorflow.keras import Model


class BaseLoss(tf.losses.Loss):
    """ Base class for common loss functions and VGG feature extraction """

    def __init__(self, img_height, img_width):
        super().__init__()
        vgg = VGG19(include_top=False, weights='imagenet', input_shape=(img_height, img_width, 3))
        vgg.trainable = False
        self.vgg_extract = Model(inputs=vgg.input, outputs=vgg.get_layer('block3_conv3').output)
        self.vgg_mean = tf.constant([123.68, 116.779, 103.939], dtype=tf.float32)

    def charbonnier_loss(self, y_true, y_pred):
        return tf.reduce_mean(tf.sqrt(tf.square(y_true - y_pred) + 1e-3 ** 2))

    def perceptual_loss(self, y_true, y_pred):
        y_t, y_p = y_true * 255.0 - self.vgg_mean, y_pred * 255.0 - self.vgg_mean
        f_t, f_p = self.vgg_extract(y_t), self.vgg_extract(y_p)
        return tf.reduce_mean(tf.square(tf.cast(f_t, tf.float32) - tf.cast(f_p, tf.float32)))


class AdvancedLoss(BaseLoss):
    """ Standard training loss combining Charbonnier, SSIM, Perceptual, and Cosine Color loss """

    def __init__(self, img_height, img_width):
        super().__init__(img_height, img_width)
        self.w_char, self.w_ssim, self.w_per, self.w_col = 1.0, 1.0, 2e-5, 0.5

    def color_loss(self, y_true, y_pred):
        """ Cosine similarity based color loss """
        y_t = tf.nn.l2_normalize(y_true + 1e-7, axis=-1)
        y_p = tf.nn.l2_normalize(y_pred + 1e-7, axis=-1)
        return tf.reduce_mean(1.0 - tf.reduce_sum(y_t * y_p, axis=-1))

    def call(self, y_true, y_pred):
        y_t, y_p = tf.cast(y_true, tf.float32), tf.cast(y_pred, tf.float32)
        l_c = self.charbonnier_loss(y_t, y_p)
        l_s = 1.0 - tf.reduce_mean(tf.image.ssim(y_t, y_p, max_val=1.0))
        l_p = self.perceptual_loss(y_t, y_p)
        l_col = self.color_loss(y_t, y_p)

        total = (self.w_char * l_c + self.w_ssim * l_s + self.w_per * l_p + self.w_col * l_col)
        return total, l_c, l_s, l_p, l_col


class FineTuneLoss(BaseLoss):
    """ Fine-tuning loss focusing on color accuracy using YUV chrominance error """

    def __init__(self, img_height, img_width):
        super().__init__(img_height, img_width)
        self.w_char, self.w_ssim, self.w_per, self.w_col = 1.0, 0.1, 2e-5, 20.0

    def color_loss(self, y_true, y_pred):
        """ YUV chrominance error for precise color alignment """
        y_true_yuv = tf.image.rgb_to_yuv(y_true)
        y_pred_yuv = tf.image.rgb_to_yuv(y_pred)
        return tf.reduce_mean(tf.abs(y_true_yuv[..., 1:] - y_pred_yuv[..., 1:]))

    def call(self, y_true, y_pred):
        y_t, y_p = tf.cast(y_true, tf.float32), tf.cast(y_pred, tf.float32)
        l_c = self.charbonnier_loss(y_t, y_p)
        l_s = 1.0 - tf.reduce_mean(tf.image.ssim(y_t, y_p, max_val=1.0))
        l_p = self.perceptual_loss(y_t, y_p)
        l_col = self.color_loss(y_t, y_p)

        total = (self.w_char * l_c + self.w_ssim * l_s + self.w_per * l_p + self.w_col * l_col)
        return total, l_c, l_s, l_p, l_col