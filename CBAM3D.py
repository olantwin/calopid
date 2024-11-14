from tensorflow.keras.layers import Layer, GlobalAveragePooling2D, GlobalMaxPooling2D, Reshape, Dense, Add, Multiply, Concatenate, Conv2D
from tensorflow.keras import layers
import tensorflow as tf
class CBAM(layers.Layer):
    def __init__(self, ratio=8, name=None):
        super(CBAM, self).__init__(name=name)
        self.ratio = ratio

    def build(self, input_shape):
        filters = input_shape[-1]

        self.global_avg_pool = GlobalAveragePooling2D()
        self.global_max_pool = GlobalMaxPooling2D()
        self.reshape = Reshape((1, filters))
        self.dense1 = Dense(filters // self.ratio, activation="relu")
        self.dense2 = Dense(filters, activation="sigmoid")

        self.conv = Conv2D(1, kernel_size=7, padding="same", activation="sigmoid")

    def call(self, input_tensor):

        avg_pool = self.global_avg_pool(input_tensor)
        max_pool = self.global_max_pool(input_tensor)
        
        avg_pool = self.reshape(avg_pool)
        max_pool = self.reshape(max_pool)
        
        avg_pool = self.dense1(avg_pool)
        max_pool = self.dense1(max_pool)
        
        channel_attention = Add()([avg_pool, max_pool])
        channel_attention = self.dense2(channel_attention)
        
        x = Multiply()([input_tensor, channel_attention])

        avg_pool = tf.reduce_mean(x, axis=-1, keepdims=True)
        max_pool = tf.reduce_max(x, axis=-1, keepdims=True)
        
        spatial_attention = Concatenate(axis=-1)([avg_pool, max_pool])
        spatial_attention = self.conv(spatial_attention)
        
        x = Multiply()([x, spatial_attention])

        return x