import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.layers import (
    Add,
    Concatenate,
    Conv1D,
    Dense,
    GlobalAveragePooling1D,
    GlobalMaxPooling1D,
    Multiply,
    Reshape,
)


class CBAM(layers.Layer):
    """
    Convolutional Block Attention Module (CBAM) layer.

    CBAM sequentially applies channel and spatial attention mechanisms to the input tensor.
    This enhances feature representations by focusing on the most informative parts of the input.
    """

    def __init__(self, ratio=8, name=None, **kwargs):
        """
        Initializes the CBAM layer.
        """
        super(CBAM, self).__init__(name=name, **kwargs)
        self.ratio = ratio

    def build(self, input_shape):
        """
        Builds the layer by initializing its weights and submodules.
        """
        filters = input_shape[-1]

        self.global_avg_pool = GlobalAveragePooling1D()
        self.global_max_pool = GlobalMaxPooling1D()
        self.reshape = Reshape((1, filters))
        self.dense1 = Dense(filters // self.ratio, activation="relu")
        self.dense2 = Dense(filters, activation="sigmoid")

        self.conv = Conv1D(1, kernel_size=7, padding="same", activation="sigmoid")

        self.built = True

    def call(self, input_tensor):
        """
        Applies the CBAM mechanism to the input tensor.
        """
        # Channel attention
        avg_pool = self.global_avg_pool(input_tensor)
        max_pool = self.global_max_pool(input_tensor)

        avg_pool = self.reshape(avg_pool)
        max_pool = self.reshape(max_pool)

        avg_pool = self.dense1(avg_pool)
        max_pool = self.dense1(max_pool)

        channel_attention = Add()([avg_pool, max_pool])
        channel_attention = self.dense2(channel_attention)

        x = Multiply()([input_tensor, channel_attention])

        # Spatial attention
        avg_pool = tf.reduce_mean(x, axis=-1, keepdims=True)
        max_pool = tf.reduce_max(x, axis=-1, keepdims=True)

        spatial_attention = Concatenate(axis=-1)([avg_pool, max_pool])
        spatial_attention = self.conv(spatial_attention)

        x = Multiply()([x, spatial_attention])

        return x

    def get_config(self):
        """
        Returns the configuration of the CBAM layer for serialization.
        """
        config = super(CBAM, self).get_config()
        config.update({"ratio": self.ratio})
        return config

    @classmethod
    def from_config(cls, config):
        """
        Creates a CBAM layer instance from its configuration.
        """
        return cls(**config)
