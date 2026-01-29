import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.layers import Input, Conv2D, Concatenate, MaxPooling2D, UpSampling2D, GlobalAveragePooling2D, \
    Multiply, Reshape


# =================NIR Feature Extraction====================
def nir_branch(input_shape=(None, None, 1), c=64):
    inp = Input(shape=input_shape)
    x = Conv2D(c, 3, padding='same', activation='relu', use_bias=True)(inp)
    x = Conv2D(c, 3, padding='same', activation='relu', use_bias=True)(x)
    return Model(inp, x, name="nir_branch")


# =================RGB Feature Extraction====================
class EDSRBlock(layers.Layer):
    def __init__(self, n_feats, res_scale=0.1):
        super().__init__()
        self.c1 = Conv2D(n_feats, 3, padding='same', use_bias=True)
        self.act = layers.ReLU()
        self.c2 = Conv2D(n_feats, 3, padding='same', use_bias=True)
        self.res_scale = res_scale

    def call(self, x):
        y = self.c1(x)
        y = self.act(y)
        y = self.c2(y)
        return x + self.res_scale * y


class EDSR(Model):
    def __init__(self, n_feats=64, n_blocks=16, res_scale=0.1):
        super().__init__()
        self.head = Conv2D(n_feats, 3, padding='same', activation='relu', use_bias=True)
        self.blocks = [EDSRBlock(n_feats, res_scale=res_scale) for _ in range(n_blocks)]
        self.tail = Conv2D(n_feats, 3, padding='same', use_bias=True)

    def call(self, x):
        x = self.head(x)
        res = x
        for blk in self.blocks:
            x = blk(x)
        x = self.tail(x)
        return x + res


# ================= SFA MODULE ====================
class SFA(layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.epsilon = 1e-6

    def call(self, content, style):
        c_mean, c_var = tf.nn.moments(content, axes=[1, 2], keepdims=True)
        c_std = tf.sqrt(c_var + self.epsilon)

        s_mean, s_var = tf.nn.moments(style, axes=[1, 2], keepdims=True)
        s_std = tf.sqrt(s_var + self.epsilon)

        color_weight = tf.nn.sigmoid(s_mean)

        normalized = (content - c_mean) / c_std
        denormalized = normalized * s_std + s_mean

        return denormalized * color_weight + denormalized


# =================ACC MODULE====================
class CBAM(layers.Layer):
    def __init__(self, reduction_ratio=16, **kwargs):
        super().__init__(**kwargs)
        self.reduction_ratio = reduction_ratio

    def build(self, input_shape):
        ch = input_shape[-1]
        self.mlp_shared_1 = layers.Dense(ch // self.reduction_ratio, activation='relu', use_bias=False)
        self.mlp_shared_2 = layers.Dense(ch, use_bias=False)

        self.spatial_conv = layers.Conv2D(1, 7, padding='same', activation='sigmoid', use_bias=False)
        super().build(input_shape)

    def call(self, x):
        avg_pool = GlobalAveragePooling2D()(x)
        max_pool = tf.reduce_max(x, axis=[1, 2])

        avg_out = self.mlp_shared_2(self.mlp_shared_1(avg_pool))
        max_out = self.mlp_shared_2(self.mlp_shared_1(max_pool))

        ch_attn = tf.nn.sigmoid(avg_out + max_out)
        ch_attn = Reshape((1, 1, -1))(ch_attn)

        x_ch = Multiply()([x, ch_attn])

        avg_pool_s = tf.reduce_mean(x_ch, axis=-1, keepdims=True)
        max_pool_s = tf.reduce_max(x_ch, axis=-1, keepdims=True)
        concat_s = Concatenate(axis=-1)([avg_pool_s, max_pool_s])

        spa_attn = self.spatial_conv(concat_s)
        output = Multiply()([x_ch, spa_attn])

        return output

class ACC(Model):
    def __init__(self, n_feats=64):
        super().__init__()
        self.conv1 = Conv2D(n_feats, 3, padding='same')
        self.lrelu1 = layers.LeakyReLU(alpha=0.2)

        self.conv2 = Conv2D(n_feats, 3, padding='same')
        self.lrelu2 = layers.LeakyReLU(alpha=0.2)

        self.cbam = CBAM(reduction_ratio=8)

        self.correction_map = Conv2D(n_feats, 1, padding='same')

        self.conv3 = Conv2D(n_feats, 3, padding='same')
        self.lrelu3 = layers.LeakyReLU(alpha=0.2)

        self.out_conv = Conv2D(n_feats, 3, padding='same')

    def call(self, x):
        residual = x

        feat = self.lrelu1(self.conv1(x))
        feat = self.lrelu2(self.conv2(feat))

        attn_feat = self.cbam(feat)

        delta = self.correction_map(attn_feat)

        x_corrected = x + delta

        out = self.lrelu3(self.conv3(x_corrected))
        out = self.out_conv(out)

        return out + residual


# ================= Sobel  ====================
class SobelEdgeLayer(layers.Layer):
    def call(self, inputs):
        edges = tf.image.sobel_edges(inputs)
        gx = edges[..., 0]
        gy = edges[..., 1]
        edge_map = tf.sqrt(gx * gx + gy * gy + 1e-6)
        return edge_map


# ==================NIR_EGCC core===================
class NIR_EGCC(Model):
    def __init__(self, n_feats=64, n_blocks=8, res_scale=0.1):
        super().__init__()

        self.sobel = SobelEdgeLayer()

        # ---  edge guidance ---
        self.edge_pre = Conv2D(n_feats, 3, padding='same', activation='relu')
        self.edge_down1 = Conv2D(n_feats, 3, strides=2, padding='same', activation='relu')
        self.edge_down2 = Conv2D(n_feats, 3, strides=2, padding='same', activation='relu')

        # ---  U-shape encoder ---
        self.enc1 = Conv2D(n_feats, 3, padding='same', activation='relu')
        self.pool1 = MaxPooling2D(2, 2)
        self.enc2 = Conv2D(n_feats * 2, 3, padding='same', activation='relu')
        self.pool2 = MaxPooling2D(2, 2)
        self.enc3 = Conv2D(n_feats * 4, 3, padding='same', activation='relu')
        self.pool3 = MaxPooling2D(2, 2)

        self.bottleneck = Conv2D(n_feats * 4, 3, padding='same', activation='relu')

        # ---  U-shape encoder ---
        self.upconv1 = UpSampling2D(size=(2, 2))
        self.dec1 = Conv2D(n_feats * 4, 3, padding='same', activation='relu')  # output: 64*4

        self.upconv2 = UpSampling2D(size=(2, 2))
        self.dec2 = Conv2D(n_feats * 2, 3, padding='same', activation='relu')  # output: 64*2

        self.upconv3 = UpSampling2D(size=(2, 2))
        self.dec3 = Conv2D(n_feats, 3, padding='same', activation='relu')  # output: 64

        # --- other module ---
        self.rgb_edsr = EDSR(n_feats, n_blocks, res_scale)
        self.nir_b = nir_branch((None, None, 1), c=n_feats)

        self.fuse = Conv2D(n_feats, 1, padding='same', activation='relu')
        self.sfa = SFA()
        self.acc = ACC(n_feats=n_feats)
        self.out_conv = Conv2D(3, 1, activation='sigmoid')  # 输出 RGB

    def call(self, inputs):
        rgb = inputs[..., :3]
        nir = inputs[..., 3:]

        Fn = self.nir_b(nir)
        edge_map = self.sobel(nir)

        fe0 = self.edge_pre(edge_map)
        fe1 = self.edge_down1(fe0)
        fe2 = self.edge_down2(fe1)

        Fr = self.rgb_edsr(rgb)

        x1 = self.enc1(Fr)
        x2 = self.pool1(x1)
        x3 = self.enc2(x2)
        x4 = self.pool2(x3)
        x5 = self.enc3(x4)
        x6 = self.pool3(x5)

        bn = self.bottleneck(x6)

        u1 = self.upconv1(bn)
        s1 = Concatenate()([u1, x5, fe2])
        d1 = self.dec1(s1)

        u2 = self.upconv2(d1)
        s2 = Concatenate()([u2, x3, fe1])
        d2 = self.dec2(s2)

        u3 = self.upconv3(d2)
        s3 = Concatenate()([u3, x1, fe0])
        d3 = self.dec3(s3)

        F_fused = self.fuse(Concatenate()([d3, Fn]))

        x = self.sfa(F_fused, d3)

        x = self.acc(x)

        return self.out_conv(x)