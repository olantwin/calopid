import tensorflow as tf
from einops import repeat, rearrange
from tensorflow.keras import layers
from cvt_modules import ConvAttention, PreLayerNorm, FeedForward
import numpy as np

from tensorflow.keras import mixed_precision
policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_global_policy(policy)

class Transformer(layers.Layer):
    """
    Transformer class that represents a transformer model layer with multiple attention and feed-forward layers.

    Attributes:
        dim (int): The dimension of the input.
        length (int): The length of the input.
        width (int): The width of the input.
        depth (int): The number of layers in the transformer.
        heads (int): The number of attention heads.
        dim_head (int): The dimension of each attention head.
        mlp_dim (int): The dimension of the feed-forward layer.
        dropout (float): The dropout rate.
        last_stage (bool): Whether this is the last stage of the model.
    """
    def __init__(self, dim, length, width, depth, heads, dim_head, mlp_dim, dropout=0., last_stage=False, **kwargs):
        """
        Initializes the Transformer layer.

        Args:
            dim (int): The dimension of the input.
            length (int): The length of the input.
            width (int): The width of the input.
            depth (int): The number of layers in the transformer.
            heads (int): The number of attention heads.
            dim_head (int): The dimension of each attention head.
            mlp_dim (int): The dimension of the feed-forward layer.
            dropout (float, optional): The dropout rate. Defaults to 0.
            last_stage (bool, optional): Whether this is the last stage of the model. Defaults to False.
            **kwargs: Additional keyword arguments.
        """
        super().__init__(**kwargs)
        self.dim = dim
        self.length = length
        self.width = width
        self.depth = depth
        self.heads = heads
        self.dim_head = dim_head
        self.mlp_dim = mlp_dim
        self.dropout = dropout
        self.last_stage = last_stage

        self.layers = []

    def build(self, input_shape):
        """
        Initialize all necessary weights and layers for the Transformer layer.
        This is called when the layer is added to the model and the input shape is known.

        Args:
            input_shape (tuple): The shape of the input tensor.
        """
        for _ in range(self.depth):
            self.layers.append([
                PreLayerNorm(fn=ConvAttention(dim=self.dim, length=self.length, width=self.width, heads=self.heads,
                                         dim_head=self.dim_head, dropout=self.dropout, last_stage=self.last_stage)),
                PreLayerNorm(fn=FeedForward(dim=self.dim, hidden_dim=self.mlp_dim, dropout=self.dropout))
            ])

    def call(self, x, training=False):
        """
        Forward pass of the Transformer layer.

        Args:
            x (tf.Tensor): The input tensor.
            training (bool, optional): Whether the layer should behave in training mode. Defaults to False.

        Returns:
            tf.Tensor: The output tensor after applying attention and feed-forward layers.
        """
        for attn, ff in self.layers:
            # Apply attention and residual connection
            x = attn(x, training=training) + x
            # Apply feed-forward and residual connection
            x = ff(x, training=training) + x
        return x

    def get_config(self):
        """
        Returns the configuration of the Transformer layer.

        Returns:
            dict: The configuration dictionary.
        """
        config = super(Transformer, self).get_config()
        config.update({
            'dim': self.dim,
            'length': self.length,
            'width': self.width,
            'depth': self.depth,
            'heads': self.heads,
            'dim_head': self.dim_head,
            'mlp_dim': self.mlp_dim,
            'dropout': self.dropout,
            'last_stage': self.last_stage
        })
        return config

    @classmethod
    def from_config(cls, config):
        """
        Creates a Transformer layer from the given configuration.

        Args:
            config (dict): The configuration dictionary.

        Returns:
            Transformer: The Transformer layer.
        """
        return cls(**config)

class RearrangeLayer(layers.Layer):
    """
    Rearrange class that manipulates tensor dimensions with optional compression.

    Attributes:
        dim (int): The dimensionality of the input tensor's channels.
        length (int): The length of the spatial dimensions of the tensor.
        width (int): The width of the spatial dimensions of the tensor.
        compression (bool): Whether to compress the tensor before rearranging.
    """
    def __init__(self, dim, length, width, compression=False, **kwargs):
        """
        Initializes the Rearrange layer.

        Args:
            dim (int): The dimensionality of the input tensor's channels.
            length (int): The length of the spatial dimensions of the tensor.
            width (int): The width of the spatial dimensions of the tensor.
            compression (bool, optional): Whether to compress the tensor before rearranging.
                Defaults to False.
            **kwargs: Additional keyword arguments.
        """
        super().__init__(**kwargs)
        self.dim = dim
        self.length = length
        self.width = width
        self.compression = compression
        
    def call(self, x):
        """
        Forward pass of the Rearrange layer.

        Rearranges the input tensor dimensions with optional compression.

        Args:
            x (tf.Tensor): The input tensor to rearrange.

        Returns:
            tf.Tensor: The rearranged tensor.
        """
        if self.compression:
            x = rearrange(x, 'b l w c -> b (l w) c', l=self.length, w=self.width)
        else:
            x = rearrange(x, 'b (l w) c -> b l w c', l=self.length, w=self.width)
        return x

    def get_config(self):
        """
        Returns the configuration of the Rearrange layer.

        Returns:
            dict: The configuration dictionary.
        """
        config = super(RearrangeLayer, self).get_config()
        config.update({
            'dim': self.dim,
            'length': self.length,
            'width': self.width,
            'compression': self.compression,
        })
        return config

    @classmethod
    def from_config(cls, config):
        """
        Creates a Rearrange layer from the given configuration.

        Args:
            config (dict): The configuration dictionary.

        Returns:
            Rearrange: An instance of the Rearrange layer.
        """
        return cls(**config)

class LastStage(layers.Layer):
    """
    LastStage class that represents the final stage of the model.

    Attributes:
        dim (int): The dimension of the input.
        batch (int): The batch size.
    """
    def __init__(self, dim, batch_size, **kwargs):
        """
        Initializes the LastStage layer.

        Args:
            dim (int): The dimension of the input.
            batch_size (int): The batch size.
            **kwargs: Additional keyword arguments.
        """
        super().__init__(**kwargs)
        self.dim = dim
        self.batch_size = batch_size

    def build(self, input_shape):
        """
        Builds the LastStage layer.

        Args:
            input_shape (tf.TensorShape): The shape of the input tensor.
        """
        self.cls_token = tf.Variable(tf.random.normal([1, 1, self.dim], dtype=tf.float16), trainable=True)

    def call(self, x):
        """
        Forward pass of the LastStage layer.

        Args:
            x (tf.Tensor): The input tensor.

        Returns:
            tf.Tensor: The output tensor with the class token appended.
        """
        b = self.batch_size
        cls_tokens = tf.tile(self.cls_token, [b, 1, 1])
        x = tf.concat([cls_tokens, x], axis=1)
        return x
    
    def get_config(self):
        """
        Returns the configuration of the LastStage layer.

        Returns:
            dict: The configuration dictionary.
        """
        config = super(LastStage, self).get_config()
        config.update({
            "dim": self.dim,
            "batch_size": self.batch_size
        })
        return config

    @classmethod
    def from_config(cls, config):
        """
        Creates a LastStage layer from the given configuration.

        Args:
            config (dict): The configuration dictionary.

        Returns:
            LastStage: The LastStage layer.
        """
        batch_size = config.get('batch_size')
        config['batch_size'] = batch_size
        return cls(**config)
    