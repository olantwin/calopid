""" Custom layers for CVT model including PreLayerNorm, SepConv2d, ConvAttention, and FeedForward. """

import tensorflow as tf
from einops import rearrange
from tensorflow.keras import layers, mixed_precision
from tensorflow.keras.layers import (
    Activation,
    BatchNormalization,
    Conv2D,
    Dense,
    DepthwiseConv2D,
    Dropout,
    LayerNormalization,
)
from tensorflow.linalg import einsum

policy = mixed_precision.Policy("mixed_float16")
mixed_precision.set_global_policy(policy)


class PreLayerNorm(layers.Layer):
    """
    Apply Layer Normalization before passing input to the given function.

    Attributes:
        fn (function): Function to be applied after normalization.
    """

    def __init__(self, fn, **kwargs):
        """
        Initialize the PreLayerNorm layer.

        Args:
            fn (function): Function to be applied after normalization.
        """
        super().__init__(**kwargs)
        self.fn = fn

    def build(self, input_shape):
        """
        Build the PreLayerNorm layer by initializing Layer Normalization.

        Args:
            input_shape (TensorShape): Shape of the input tensor.
        """
        self.norm = LayerNormalization(axis=-1)

    def call(self, x, training=False, **kwargs):
        """
        Perform forward pass for the PreLayerNorm layer.

        Args:
            x (Tensor): Input tensor.
            training (bool): Specify if call is for training or inference. Default is False.

        Returns:
            Tensor: Output tensor after applying normalization and the given function.
        """
        x = self.norm(x)
        return self.fn(x, training=training, **kwargs)

    def get_config(self):
        """
        Return the configuration of the layer.

        Returns:
            dict: Dictionary containing layer configuration.
        """
        config = super(PreLayerNorm, self).get_config()
        config.update({"fn": self.fn})
        return config

    @classmethod
    def from_config(cls, config):
        """
        Create a layer from its configuration.

        Args:
            config (dict): Dictionary containing layer configuration.

        Returns:
            PreLayerNorm: Layer object created from the config.
        """
        return cls(**config)


class SepConv2d(layers.Layer):
    """
    Implement separable convolution: depthwise conv followed by pointwise conv.

    Attributes:
        filters (int): Number of output filters.
        kernel_size (int or tuple): Convolution window size.
        stride (int or tuple): Stride size.
        padding (str): Padding method ('valid' or 'same').
        dilation (int or tuple): Dilation rate.
    """

    def __init__(
        self, filters, kernel_size, stride=1, padding="same", dilation=1, **kwargs
    ):
        """
        Initialize the SepConv2d layer.

        Args:
            filters (int): Number of output filters.
            kernel_size (int or tuple): Convolution window size.
            stride (int or tuple): Stride size.
            padding (str): Padding method.
            dilation (int or tuple): Dilation rate.
        """
        super().__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation

    def build(self, input_shape):
        """
        Build the separable convolution layer components.

        Args:
            input_shape (TensorShape): Shape of the input tensor.
        """
        self.depthconv = DepthwiseConv2D(
            kernel_size=self.kernel_size,
            strides=self.stride,
            padding=self.padding,
            dilation_rate=self.dilation,
        )
        self.batchnorm = BatchNormalization()
        self.pointconv = Conv2D(filters=self.filters, kernel_size=1)

    def call(self, x, training=False):
        """
        Perform forward pass for the SepConv2d layer.

        Args:
            x (Tensor): Input tensor.
            training (bool): Specify training mode. Default is False.

        Returns:
            Tensor: Output tensor after depthwise conv, batch norm, and pointwise conv.
        """
        x = self.depthconv(x)
        x = self.batchnorm(x, training=training)
        x = self.pointconv(x)
        return x

    def get_config(self):
        """
        Return the configuration of the layer.

        Returns:
            dict: Dictionary containing layer configuration.
        """
        config = super(SepConv2d, self).get_config()
        config.update(
            {
                "filters": self.filters,
                "kernel_size": self.kernel_size,
                "stride": self.stride,
                "padding": self.padding,
                "dilation": self.dilation,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        """
        Create a layer from its configuration.

        Args:
            config (dict): Dictionary containing layer configuration.

        Returns:
            SepConv2d: Layer object created from the config.
        """
        return cls(**config)


class ConvAttention(layers.Layer):
    """
    Implement convolutional attention mechanism using separable convolutions.

    Attributes:
        dim (int): Output dimensionality.
        length (int): Input length dimension.
        width (int): Input width dimension.
        heads (int): Number of attention heads.
        dim_head (int): Dimension per attention head.
        kernel_size (int): Kernel size of convolutions.
        q_stride (int): Stride for query conv.
        k_stride (int): Stride for key conv.
        v_stride (int): Stride for value conv.
        dropout (float): Dropout rate.
        last_stage (bool): Whether this is the last stage.
    """

    def __init__(
        self,
        dim,
        length,
        width,
        heads,
        dim_head,
        kernel_size=3,
        q_stride=1,
        k_stride=1,
        v_stride=1,
        dropout=0.0,
        last_stage=False,
        **kwargs,
    ):
        """
        Initialize the ConvAttention layer.

        Args:
            dim (int): Output dimensionality.
            length (int): Input length.
            width (int): Input width.
            heads (int): Number of attention heads.
            dim_head (int): Dimension per attention head.
            kernel_size (int, optional): Kernel size. Default 3.
            q_stride (int, optional): Query stride. Default 1.
            k_stride (int, optional): Key stride. Default 1.
            v_stride (int, optional): Value stride. Default 1.
            dropout (float, optional): Dropout rate. Default 0.0.
            last_stage (bool, optional): If last stage. Default False.
        """
        super().__init__(**kwargs)
        self.dim = dim
        self.last_stage = last_stage
        self.length = length
        self.width = width
        self.heads = heads
        self.scale = dim_head**-0.5
        self.inner_dim = dim_head * heads
        self.project_out = not (heads == 1 and dim_head == dim)
        self.kernel_size = kernel_size
        self.q_stride = q_stride
        self.k_stride = k_stride
        self.v_stride = v_stride
        self.dropout_rate = dropout

    def build(self, input_shape):
        """
        Build the ConvAttention layer components.

        Args:
            input_shape (TensorShape): Shape of the input tensor.
        """
        self.to_q = SepConv2d(
            filters=self.dim,
            kernel_size=self.kernel_size,
            stride=self.q_stride,
            padding="same",
        )
        self.to_k = SepConv2d(
            filters=self.dim,
            kernel_size=self.kernel_size,
            stride=self.k_stride,
            padding="same",
        )
        self.to_v = SepConv2d(
            filters=self.dim,
            kernel_size=self.kernel_size,
            stride=self.v_stride,
            padding="same",
        )

        if self.project_out:
            self.to_out = tf.keras.Sequential(
                [layers.Dense(self.dim), layers.Dropout(self.dropout_rate)]
            )
        else:
            self.to_out = tf.keras.layers.Lambda(lambda x: x)

    def call(self, x, training=False):
        """
        Apply the convolutional attention mechanism.

        Args:
            x (Tensor): Input tensor.
            training (bool, optional): Training mode flag. Default False.

        Returns:
            Tensor: Output tensor after attention.
        """
        _, _, _, h = *x.shape, self.heads
        if self.last_stage:
            cls_token = x[:, 0]
            x = x[:, 1:]
            cls_token = rearrange(
                tf.expand_dims(cls_token, axis=1), "b c (h d) -> b h c d", h=h
            )
        x = rearrange(x, "b (l w) c -> b l w c", l=self.length, w=self.width)

        q = self.to_q(x, training=training)
        q = rearrange(q, "b l w (h d) -> b h (l w) d", h=h)

        v = self.to_v(x, training=training)
        v = rearrange(v, "b l w (h d) -> b h (l w) d", h=h)

        k = self.to_k(x, training=training)
        k = rearrange(k, "b l w (h d) -> b h (l w) d", h=h)

        if self.last_stage:
            q = tf.concat((cls_token, q), axis=2)
            v = tf.concat((cls_token, v), axis=2)
            k = tf.concat((cls_token, k), axis=2)

        dots = einsum("b h i d, b h j d -> b h i j", q, k) * self.scale

        attn = tf.nn.softmax(dots, axis=-1)

        out = einsum("b h i j, b h j d -> b h i d", attn, v)
        out = rearrange(out, "b h c d -> b c (h d)")
        out = self.to_out(out)
        return out

    def get_config(self):
        """
        Return the configuration of the ConvAttention layer.

        Returns:
            dict: Configuration parameters.
        """
        config = super(ConvAttention, self).get_config()
        config.update(
            {
                "dim": self.dim,
                "last_stage": self.last_stage,
                "length": self.length,
                "width": self.width,
                "heads": self.heads,
                "scale": self.scale,
                "inner_dim": self.inner_dim,
                "project_out": self.project_out,
                "kernel_size": self.kernel_size,
                "q_stride": self.q_stride,
                "v_stride": self.v_stride,
                "k_stride": self.k_stride,
                "dropout_rate": self.dropout_rate,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        """
        Create a ConvAttention layer from its configuration.

        Args:
            config (dict): Configuration parameters.

        Returns:
            ConvAttention: Layer object created from the config.
        """
        return cls(**config)


class FeedForward(layers.Layer):
    """
    Implement feed-forward network with GELU activation and dropout.

    Attributes:
        dim (int): Output dimensionality.
        hidden_dim (int): Hidden layer dimensionality.
        dropout (float): Dropout rate.
    """

    def __init__(self, dim, hidden_dim, dropout=0.0, **kwargs):
        """
        Initialize the FeedForward layer.

        Args:
            dim (int): Output dimensionality.
            hidden_dim (int): Hidden layer dimensionality.
            dropout (float): Dropout rate.
        """
        super().__init__(**kwargs)
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.dropout_rate = dropout

    def build(self, input_shape):
        """
        Build the feed-forward network components.

        Args:
            input_shape (TensorShape): Shape of input tensor.
        """
        self.dense1 = Dense(self.hidden_dim)
        self.activation = Activation("gelu")
        self.dropout1 = Dropout(self.dropout_rate)
        self.dense2 = Dense(self.dim)
        self.dropout2 = Dropout(self.dropout_rate)

    def call(self, x, training=False, **kwargs):
        """
        Perform forward pass for the FeedForward layer.

        Args:
            x (Tensor): Input tensor.
            training (bool): Specify training mode. Default False.

        Returns:
            Tensor: Output tensor after feed-forward network.
        """
        x = self.dense1(x)
        x = self.activation(x)
        x = self.dropout1(x, training=training)
        x = self.dense2(x)
        x = self.dropout2(x, training=training)
        return x

    def get_config(self):
        """
        Return the configuration of the FeedForward layer.

        Returns:
            dict: Layer configuration.
        """
        config = super(FeedForward, self).get_config()
        config.update(
            {
                "dim": self.dim,
                "hidden_dim": self.hidden_dim,
                "dropout_rate": self.dropout_rate,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        """
        Create a FeedForward layer from its configuration.

        Args:
            config (dict): Layer configuration.

        Returns:
            FeedForward: Layer object created from the config.
        """
        return cls(**config)
