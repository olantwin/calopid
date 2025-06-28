"""Transformer model components including Transformer, RearrangeLayer, and LastStage layers."""

import tensorflow as tf
from einops import rearrange
from tensorflow.keras import layers, mixed_precision

from cvt_modules import ConvAttention, FeedForward, PreLayerNorm

policy = mixed_precision.Policy("mixed_float16")
mixed_precision.set_global_policy(policy)


class Transformer(layers.Layer):
    """
    Implement a Transformer with multiple layers of attention and feed-forward blocks.

    Attributes:
        dim (int): Input dimension.
        length (int): Input length.
        width (int): Input width.
        depth (int): Number of Transformer layers.
        heads (int): Number of attention heads.
        dim_head (int): Dimension per attention head.
        mlp_dim (int): Hidden dimension of feed-forward layer.
        dropout (float): Dropout rate.
        last_stage (bool): Whether this is the last stage.
    """

    def __init__(
        self,
        dim,
        length,
        width,
        depth,
        heads,
        dim_head,
        mlp_dim,
        dropout=0.0,
        last_stage=False,
        **kwargs,
    ):
        """
        Initialize the Transformer layer.

        Args:
            dim (int): Input dimension.
            length (int): Input length.
            width (int): Input width.
            depth (int): Number of Transformer layers.
            heads (int): Number of attention heads.
            dim_head (int): Dimension per attention head.
            mlp_dim (int): Hidden dimension of feed-forward layer.
            dropout (float, optional): Dropout rate. Defaults to 0.0.
            last_stage (bool, optional): Whether this is the last stage. Defaults to False.
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
        Build the Transformer layers.

        Args:
            input_shape (tuple): Shape of input tensor.
        """
        for _ in range(self.depth):
            self.layers.append(
                [
                    PreLayerNorm(
                        fn=ConvAttention(
                            dim=self.dim,
                            length=self.length,
                            width=self.width,
                            heads=self.heads,
                            dim_head=self.dim_head,
                            dropout=self.dropout,
                            last_stage=self.last_stage,
                        )
                    ),
                    PreLayerNorm(
                        fn=FeedForward(
                            dim=self.dim, hidden_dim=self.mlp_dim, dropout=self.dropout
                        )
                    ),
                ]
            )

    def call(self, x, training=False):
        """
        Perform forward pass through the Transformer layers.

        Args:
            x (tf.Tensor): Input tensor.
            training (bool, optional): Whether layer is in training mode. Defaults to False.

        Returns:
            tf.Tensor: Output tensor after attention and feed-forward layers.
        """
        for attn, ff in self.layers:
            # Apply attention and residual connection
            x = attn(x, training=training) + x
            # Apply feed-forward and residual connection
            x = ff(x, training=training) + x
        return x

    def get_config(self):
        """
        Return configuration of the Transformer layer.

        Returns:
            dict: Configuration dictionary.
        """
        config = super(Transformer, self).get_config()
        config.update(
            {
                "dim": self.dim,
                "length": self.length,
                "width": self.width,
                "depth": self.depth,
                "heads": self.heads,
                "dim_head": self.dim_head,
                "mlp_dim": self.mlp_dim,
                "dropout": self.dropout,
                "last_stage": self.last_stage,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        """
        Create a Transformer layer from configuration.

        Args:
            config (dict): Configuration dictionary.

        Returns:
            Transformer: Instantiated Transformer layer.
        """
        return cls(**config)


class RearrangeLayer(layers.Layer):
    """
    Rearrange tensor dimensions with optional compression.

    Attributes:
        dim (int): Channel dimension of input tensor.
        length (int): Spatial length dimension.
        width (int): Spatial width dimension.
        compression (bool): Whether to compress tensor before rearranging.
    """

    def __init__(self, dim, length, width, compression=False, **kwargs):
        """
        Initialize the Rearrange layer.

        Args:
            dim (int): Channel dimension of input tensor.
            length (int): Spatial length dimension.
            width (int): Spatial width dimension.
            compression (bool, optional): Whether to compress tensor. Defaults to False.
            **kwargs: Additional keyword arguments.
        """
        super().__init__(**kwargs)
        self.dim = dim
        self.length = length
        self.width = width
        self.compression = compression

    def call(self, x):
        """
        Rearrange the input tensor dimensions.

        Args:
            x (tf.Tensor): Input tensor.

        Returns:
            tf.Tensor: Rearranged tensor.
        """
        if self.compression:
            x = rearrange(x, "b l w c -> b (l w) c", l=self.length, w=self.width)
        else:
            x = rearrange(x, "b (l w) c -> b l w c", l=self.length, w=self.width)
        return x

    def get_config(self):
        """
        Return configuration of the Rearrange layer.

        Returns:
            dict: Configuration dictionary.
        """
        config = super(RearrangeLayer, self).get_config()
        config.update(
            {
                "dim": self.dim,
                "length": self.length,
                "width": self.width,
                "compression": self.compression,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        """
        Create a Rearrange layer from configuration.

        Args:
            config (dict): Configuration dictionary.

        Returns:
            RearrangeLayer: Instantiated Rearrange layer.
        """
        return cls(**config)


class LastStage(layers.Layer):
    """
    Implement the last stage of the model with class token addition.

    Attributes:
        dim (int): Dimension of input tensor.
        batch_size (int): Batch size.
    """

    def __init__(self, dim, batch_size, **kwargs):
        """
        Initialize the LastStage layer.

        Args:
            dim (int): Dimension of input tensor.
            batch_size (int): Batch size.
            **kwargs: Additional keyword arguments.
        """
        super().__init__(**kwargs)
        self.dim = dim
        self.batch_size = batch_size

    def build(self, input_shape):
        """
        Build the LastStage layer by creating class token variable.

        Args:
            input_shape (tf.TensorShape): Shape of input tensor.
        """
        self.cls_token = tf.Variable(
            tf.random.normal([1, 1, self.dim], dtype=tf.float16), trainable=True
        )

    def call(self, x):
        """
        Append class token to the input tensor.

        Args:
            x (tf.Tensor): Input tensor.

        Returns:
            tf.Tensor: Tensor with class token prepended.
        """
        b = self.batch_size
        cls_tokens = tf.tile(self.cls_token, [b, 1, 1])
        x = tf.concat([cls_tokens, x], axis=1)
        return x

    def get_config(self):
        """
        Return configuration of the LastStage layer.

        Returns:
            dict: Configuration dictionary.
        """
        config = super(LastStage, self).get_config()
        config.update({"dim": self.dim, "batch_size": self.batch_size})
        return config

    @classmethod
    def from_config(cls, config):
        """
        Create a LastStage layer from configuration.

        Args:
            config (dict): Configuration dictionary.

        Returns:
            LastStage: Instantiated LastStage layer.
        """
        batch_size = config.get("batch_size")
        config["batch_size"] = batch_size
        return cls(**config)
