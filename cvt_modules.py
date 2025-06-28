import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.layers import (
    Activation,
    BatchNormalization,
    Concatenate,
    Conv2D,
    Dense,
    DepthwiseConv2D,
    Dropout,
    Lambda,
    LayerNormalization,
)
from tensorflow.keras.activations import gelu
from tensorflow.keras import mixed_precision
from tensorflow.linalg import einsum
from einops import rearrange, repeat

from tensorflow.keras import mixed_precision
policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_global_policy(policy)


class PreLayerNorm(layers.Layer):
    """
    Pre-LayerNorm layer: Applies Layer Normalization before passing input to the given function.

    Attributes:
        fn (function): Function to be applied after normalization.
    """
    def __init__(self, fn, **kwargs):
        """
        Initializes the PreLayerNorm layer.

        Args:
            fn (function): Function to be applied after normalization.
        """
        super().__init__(**kwargs)
        self.fn = fn
        
    def build(self, input_shape):
        """
        Builds the Pre-LayerNorm layer by initializing Layer Normalization.
        
        Args:
            input_shape (TensorShape): Shape of the input tensor.
        """
        self.norm = LayerNormalization(axis=-1)

    def call(self, x, training=False, **kwargs):
        """
        Forward pass for the Pre-LayerNorm layer.
        
        Args:
            x (Tensor): Input tensor.
            training (bool): Boolean to specify if the call is for training or inference. Default is False.
        
        Returns:
            Tensor: Output tensor after applying layer normalization and the given function.
        """
        x = self.norm(x)
        return self.fn(x, training=training, **kwargs)

    def get_config(self):
        """
        Returns the configuration of the layer.
        
        Returns:
            dict: Dictionary containing layer configuration.
        """
        config = super(PreLayerNorm, self).get_config()
        config.update({
            "fn": self.fn
        })
        return config

    @classmethod
    def from_config(cls, config):
        """
        Creates a layer from its configuration.
        
        Args:
            config (dict): Dictionary containing layer configuration.
            
        Returns:
            PreLayerNorm: Pre-LayerNorm Layer object created from the config.
        """
        return cls(**config)


class SepConv2d(layers.Layer):
    """
    Separable Convolutional Layer: Depthwise convolution followed by a pointwise convolution.

    Attributes:
        filters (int): Number of output filters in the convolution.
        kernel_size (int or tuple/list of 2 ints): Dimensions of the convolution window.
        stride (int or tuple/list of 2 ints): Strides of the convolution. Default is 1.
        padding (str): Padding method to use ('valid' or 'same'). Default is 'same'.
        dilation (int or tuple/list of 2 ints): Dilation rate to use for dilated convolution. Default is 1.
    """
    def __init__(self, filters, kernel_size, stride=1, padding='same', dilation=1, **kwargs):
        """
        Initializes the SepConv2d layer.

        Args:
            filters (int): Number of output filters in the convolution.
            kernel_size (int or tuple/list of 2 ints): Dimensions of the convolution window.
            stride (int or tuple/list of 2 ints): Strides of the convolution.
            padding (str): Padding method to use ('valid' or 'same').
            dilation (int or tuple/list of 2 ints): Dilation rate to use for dilated convolution.
        """
        super().__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        
    def build(self, input_shape):
        """
        Builds the Separable Convolutional layer by initializing depthwise and pointwise convolutions.
        
        Args:
            input_shape (TensorShape): Shape of the input tensor.
        """
        self.depthconv = DepthwiseConv2D(kernel_size=self.kernel_size,
                                         strides=self.stride,
                                         padding=self.padding,
                                         dilation_rate=self.dilation)
        self.batchnorm = BatchNormalization()
        self.pointconv = Conv2D(filters=self.filters, kernel_size=1)
        
    def call(self, x, training=False):
        """
        Forward pass for the Separable Convolutional layer.
        
        Args:
            x (Tensor): Input tensor.
            training (bool): Boolean to specify if the call is for training or inference. Default is False.
        
        Returns:
            Tensor: Output tensor after applying depthwise conv, batch normalization, and pointwise conv.
        """
        x = self.depthconv(x)
        x = self.batchnorm(x, training=training)
        x = self.pointconv(x)
        return x
        
    def get_config(self):
        """
        Returns the configuration of the layer.
        
        Returns:
            dict: Dictionary containing layer configuration.
        """
        config = super(SepConv2d, self).get_config()
        config.update({
            "filters": self.filters,
            "kernel_size": self.kernel_size,
            "stride": self.stride,
            "padding": self.padding,
            "dilation": self.dilation
        })
        return config
        
    @classmethod
    def from_config(cls, config):
        """
        Creates a layer from its configuration.
        
        Args:
            config (dict): Dictionary containing layer configuration.
            
        Returns:
            SepConv2d: Separable Convolutional Layer object created from the config.
        """
        return cls(**config)        


class ConvAttention(layers.Layer):
    """
    Convolutional Attention Layer: Applies attention mechanism with separable convolutions.

    Attributes:
        dim (int): Dimensionality of the output space.
        length (int): Length of the input sequence.
        width (int): Width of the input sequence.
        heads (int): Number of attention heads.
        dim_head (int): Dimensionality of each attention head.
        kernel_size (int): Kernel size of the convolutions. Default is 3.
        q_stride (int): Stride for the query convolution. Default is 1.
        k_stride (int): Stride for the key convolution. Default is 1.
        v_stride (int): Stride for the value convolution. Default is 1.
        dropout (float): Fraction of the input units to drop. Default is 0.0.
        last_stage (bool): Flag to indicate if this is the last stage. Default is False.
    """
    def __init__(self, dim, length, width, heads, dim_head, kernel_size=3, q_stride=1, k_stride=1, v_stride=1, dropout=0., last_stage=False, **kwargs):
        """
        Initialize the ConvAttention layer.

        Args:
            dim (int): Dimensionality of the output space.
            length (int): Length of the input sequence.
            width (int): Width of the input sequence.
            heads (int): Number of attention heads.
            dim_head (int): Dimensionality of each attention head.
            kernel_size (int, optional): Kernel size of the convolutions. Default is 3.
            q_stride (int, optional): Stride for the query convolution. Default is 1.
            k_stride (int, optional): Stride for the key convolution. Default is 1.
            v_stride (int, optional): Stride for the value convolution. Default is 1.
            dropout (float, optional): Fraction of the input units to drop. Default is 0.0.
            last_stage (bool, optional): Flag to indicate if this is the last stage. Default is False.
        """
        super().__init__(**kwargs)
        self.dim = dim
        self.last_stage = last_stage
        self.length = length
        self.width = width
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.inner_dim = dim_head * heads
        self.project_out = not (heads == 1 and dim_head == dim)
        self.kernel_size = kernel_size
        self.q_stride = q_stride
        self.k_stride = k_stride
        self.v_stride = v_stride
        self.dropout_rate = dropout

    def build(self, input_shape):
        """
        Build the ConvAttention layer.

        Args:
            input_shape (TensorShape): Shape of the input tensor.
        """
        self.to_q = SepConv2d(filters=self.dim, kernel_size=self.kernel_size, stride=self.q_stride, padding='same')
        self.to_k = SepConv2d(filters=self.dim, kernel_size=self.kernel_size, stride=self.k_stride, padding='same')
        self.to_v = SepConv2d(filters=self.dim, kernel_size=self.kernel_size, stride=self.v_stride, padding='same')

        if self.project_out:
            self.to_out = tf.keras.Sequential([
                layers.Dense(self.dim),
                layers.Dropout(self.dropout_rate)
            ])
        else:
            self.to_out = tf.keras.layers.Lambda(lambda x: x)
        
    def call(self, x, training=False):
        """
        Apply the ConvAttention mechanism.

        Args:
            x (Tensor): Input tensor.
            training (bool, optional): Whether the layer is in training mode. Default is False.

        Returns:
            Tensor: Output tensor after applying the attention mechanism.
        """
        b, _, c, h = *x.shape, self.heads
        if self.last_stage:
            cls_token = x[:, 0]
            x = x[:, 1:]
            cls_token = rearrange(tf.expand_dims(cls_token, axis=1), 'b c (h d) -> b h c d', h=h)
        x = rearrange(x, 'b (l w) c -> b l w c', l=self.length, w=self.width)
        
        q = self.to_q(x, training=training)
        q = rearrange(q, 'b l w (h d) -> b h (l w) d', h=h)

        v = self.to_v(x, training=training)
        v = rearrange(v, 'b l w (h d) -> b h (l w) d', h=h)

        k = self.to_k(x, training=training)
        k = rearrange(k, 'b l w (h d) -> b h (l w) d', h=h)

        if self.last_stage:
            q = tf.concat((cls_token, q), axis=2)
            v = tf.concat((cls_token, v), axis=2)
            k = tf.concat((cls_token, k), axis=2)

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        attn = tf.nn.softmax(dots, axis=-1)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h c d -> b c (h d)')
        out =  self.to_out(out)
        return out

    def get_config(self):
        """
        Returns the configuration of the ConvAttention layer.

        Returns:
            dict: Dictionary containing the configuration parameters.
        """
        config = super(ConvAttention, self).get_config()
        config.update({
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
            "dropout_rate": self.dropout_rate
        })
        return config

    @classmethod
    def from_config(cls, config):
        """
        Creates an instance of ConvAttention from the given configuration.

        Args:
            config (dict): Dictionary containing the configuration parameters.

        Returns:
            ConvAttention: An instance of the ConvAttention class.
        """
        return cls(**config)


class FeedForward(layers.Layer):
    """
    FeedForward Layer: Applies dense layers with GELU activation and dropout.

    Attributes:
        dim (int): Dimensionality of the output space.
        hidden_dim (int): Dimensionality of the hidden layer.
        dropout (float): Fraction of the input units to drop. Default is 0.0.
    """
    def __init__(self, dim, hidden_dim, dropout=0., **kwargs):
        """
        Initializes the FeedForward layer.

        Args:
            dim (int): Dimensionality of the output space.
            hidden_dim (int): Dimensionality of the hidden layer.
            dropout (float): Fraction of the input units to drop. Default is 0.0.
        """
        super().__init__(**kwargs)
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.dropout_rate = dropout
        
    def build(self, input_shape):
        """
        Builds the FeedForward layer by initializing dense layers, activation, and dropout.
        
        Args:
            input_shape (TensorShape): Shape of the input tensor.
        """
        self.dense1 = Dense(self.hidden_dim)
        self.activation = Activation('gelu')
        self.dropout1 = Dropout(self.dropout_rate)
        self.dense2 = Dense(self.dim)
        self.dropout2 = Dropout(self.dropout_rate)

    def call(self, x, training=False, **kwargs):
        """
        Forward pass for the FeedForward layer.
        
        Args:
            x (Tensor): Input tensor.
            training (bool): Boolean to specify if the call is for training or inference. Default is False.
        
        Returns:
            Tensor: Output tensor after applying dense layers, activation, and dropout.
        """
        x = self.dense1(x)
        x = self.activation(x)
        x = self.dropout1(x, training=training)
        x = self.dense2(x)
        x = self.dropout2(x, training=training)
        return x
        
    def get_config(self):
        """
        Returns the configuration of the layer.
        
        Returns:
            dict: Dictionary containing layer configuration.
        """
        config = super(FeedForward, self).get_config()
        config.update({
            "dim": self.dim,
            "hidden_dim": self.hidden_dim,
            "dropout_rate": self.dropout_rate
        })
        return config

    @classmethod
    def from_config(cls, config):
        """
        Creates a layer from its configuration.
        
        Args:
            config (dict): Dictionary containing layer configuration.
            
        Returns:
            FeedForward: FeedForward Layer object created from the config.
        """
        return cls(**config)