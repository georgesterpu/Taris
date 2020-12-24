import tensorflow as tf
from absl import flags

FLAGS = flags.FLAGS


class Attention(tf.keras.layers.Layer):
    """Multi-headed attention layer."""

    def __init__(self, hidden_size, num_heads, attention_dropout):
        """Initialize Attention.

        Args:
          hidden_size: int, output dim of hidden layer.
          num_heads: int, number of heads to repeat the same attention structure.
          attention_dropout: float, dropout rate inside attention for training.
        """
        if hidden_size % num_heads:
            raise ValueError(
                "Hidden size ({}) must be divisible by the number of heads ({})."
                    .format(hidden_size, num_heads))

        super(Attention, self).__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.attention_dropout = attention_dropout

    def build(self, input_shape):
        """Builds the layer."""
        # Layers for linearly projecting the queries, keys, and values.
        size_per_head = self.hidden_size // self.num_heads
        self.query_dense_layer = Dense3D(
            self.num_heads, size_per_head, kernel_initializer="glorot_uniform",
            use_bias=False, name="query")
        self.key_dense_layer = Dense3D(
            self.num_heads, size_per_head, kernel_initializer="glorot_uniform",
            use_bias=False, name="key")
        self.value_dense_layer = Dense3D(
            self.num_heads, size_per_head, kernel_initializer="glorot_uniform",
            use_bias=False, name="value")
        self.output_dense_layer = Dense3D(
            self.num_heads, size_per_head, kernel_initializer="glorot_uniform",
            use_bias=False, output_projection=True, name="output_transform")
        super(Attention, self).build(input_shape)

    def get_config(self):
        return {
            "hidden_size": self.hidden_size,
            "num_heads": self.num_heads,
            "attention_dropout": self.attention_dropout,
        }

    def call(self, query_input, source_input, bias, training, cache=None):
        """Apply attention mechanism to query_input and source_input.

        Args:
          query_input: A tensor with shape [batch_size, length_query, hidden_size].
          source_input: A tensor with shape [batch_size, length_source, hidden_size].
          bias: A tensor with shape [batch_size, 1, length_query, length_source],
            the attention bias that will be added to the result of the dot product.
          training: A bool, whether in training mode or not.
          cache: (Used during prediction) A dictionary with tensors containing
            results of previous attentions. The dictionary must have the items:
                {"k": tensor with shape [batch_size, i, heads, dim_per_head],
                 "v": tensor with shape [batch_size, i, heads, dim_per_head]}
            where i is the current decoded length

        Returns:
          Attention layer output with shape [batch_size, length_query, hidden_size]
        """
        # Linearly project the query, key and value using different learned
        # projections. Splitting heads is automatically done during the linear
        # projections --> [batch_size, length, num_heads, dim_per_head].
        query = self.query_dense_layer(query_input)
        key = self.key_dense_layer(source_input)
        value = self.value_dense_layer(source_input)

        if cache is not None:
            # Combine cached keys and values with new keys and values.
            key = tf.concat([tf.cast(cache["k"], key.dtype), key], axis=1)
            value = tf.concat([tf.cast(cache["v"], value.dtype), value], axis=1)

            # Update cache
            cache["k"] = key
            cache["v"] = value

        # Scale query to prevent the dot product between query and key from growing
        # too large.
        depth = (self.hidden_size // self.num_heads)
        query *= depth ** -0.5

        # Calculate dot product attention
        logits = tf.einsum("BTNH,BFNH->BNFT", key, query)
        logits += bias
        # Note that softmax internally performs math operations using float32
        # for numeric stability. When training with float16, we keep the input
        # and output in float16 for better performance.
        weights = tf.nn.softmax(logits, name="attention_weights")
        weights = tf.where(bias > -1e9, weights, tf.zeros_like(weights))
        if FLAGS.plot_alignments is True:
            tf.summary.image(name=self.name, data=tf.expand_dims(tf.squeeze(weights, 1), -1), step=1, max_outputs=1)
        if training:
            weights = tf.nn.dropout(weights, rate=self.attention_dropout)
        attention_output = tf.einsum("BNFT,BTNH->BFNH", weights, value)

        # Run the outputs through another linear projection layer. Recombining heads
        # is automatically done --> [batch_size, length, hidden_size]
        attention_output = self.output_dense_layer(attention_output)
        return attention_output


class SelfAttention(Attention):
    """Multiheaded self-attention layer."""

    def call(self, query_input, bias, training, cache=None):
        return super(SelfAttention, self).call(
            query_input, query_input, bias, training, cache)


class Dense3D(tf.keras.layers.Layer):
    """A Dense Layer using 3D kernel with tf.einsum implementation.

    Attributes:
      num_attention_heads: An integer, number of attention heads for each
        multihead attention layer.
      size_per_head: An integer, hidden size per attention head.
      hidden_size: An integer, dimension of the hidden layer.
      kernel_initializer: An initializer for the kernel weight.
      bias_initializer: An initializer for the bias.
      activation: An activation function to use. If nothing is specified, no
        activation is applied.
      use_bias: A bool, whether the layer uses a bias.
      output_projection: A bool, whether the Dense3D layer is used for output
        linear projection.
      backward_compatible: A bool, whether the variables shape are compatible with
        checkpoints converted from TF 1.x.
    """

    def __init__(self,
                 num_attention_heads=12,
                 size_per_head=72,
                 kernel_initializer=None,
                 bias_initializer="zeros",
                 activation=None,
                 use_bias=True,
                 output_projection=False,
                 **kwargs):
        """Inits Dense3D."""
        super(Dense3D, self).__init__(**kwargs)
        self.num_attention_heads = num_attention_heads
        self.size_per_head = size_per_head
        self.hidden_size = num_attention_heads * size_per_head
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer
        self.activation = activation
        self.use_bias = use_bias
        self.output_projection = output_projection

    @property
    def kernel_shape(self):
        if self.output_projection:
            return [self.num_attention_heads, self.size_per_head, self.hidden_size]
        return [self.last_dim, self.num_attention_heads, self.size_per_head]

    @property
    def bias_shape(self):
        if self.output_projection:
            return [self.hidden_size]
        return [self.num_attention_heads, self.size_per_head]

    def build(self, input_shape):
        """Implements build() for the layer."""
        dtype = tf.as_dtype(self.dtype or tf.keras.backend.floatx())
        if not (dtype.is_floating or dtype.is_complex):
            raise TypeError("Unable to build `Dense3D` layer with non-floating "
                            "point (and non-complex) dtype %s" % (dtype,))
        input_shape = tf.TensorShape(input_shape)
        if tf.compat.dimension_value(input_shape[-1]) is None:
            raise ValueError("The last dimension of the inputs to `Dense3D` "
                             "should be defined. Found `None`.")
        self.last_dim = tf.compat.dimension_value(input_shape[-1])
        self.input_spec = tf.keras.layers.InputSpec(
            min_ndim=3, axes={-1: self.last_dim})
        # Determines variable shapes.
        kernel_shape = self.kernel_shape
        bias_shape = self.bias_shape

        self.kernel = self.add_weight(
            "kernel",
            shape=kernel_shape,
            initializer=self.kernel_initializer,
            dtype=self.dtype,
            trainable=True)
        if self.use_bias:
            self.bias = self.add_weight(
                "bias",
                shape=bias_shape,
                initializer=self.bias_initializer,
                dtype=self.dtype,
                trainable=True)
        else:
            self.bias = None
        super(Dense3D, self).build(input_shape)

    def call(self, inputs):
        """Implements ``call()`` for Dense3D.

        Args:
          inputs: A float tensor of shape [batch_size, sequence_length, hidden_size]
            when output_projection is False, otherwise a float tensor of shape
            [batch_size, sequence_length, num_heads, dim_per_head].

        Returns:
          The projected tensor with shape [batch_size, sequence_length, num_heads,
            dim_per_head] when output_projection is False, otherwise [batch_size,
            sequence_length, hidden_size].
        """
        kernel = self.kernel
        bias = self.bias

        if self.output_projection:
            ret = tf.einsum("abcd,cde->abe", inputs, kernel)
        else:
            ret = tf.einsum("abc,cde->abde", inputs, kernel)
        if self.use_bias:
            ret += bias
        if self.activation is not None:
            return self.activation(ret)
        return ret


class FeedForwardNetwork(tf.keras.layers.Layer):
    """Fully connected feedforward network."""

    def __init__(self, hidden_size, filter_size, relu_dropout):
        """Initialize FeedForwardNetwork.

        Args:
          hidden_size: int, output dim of hidden layer.
          filter_size: int, filter size for the inner (first) dense layer.
          relu_dropout: float, dropout rate for training.
        """
        super(FeedForwardNetwork, self).__init__()
        self.hidden_size = hidden_size
        self.filter_size = filter_size
        self.relu_dropout = relu_dropout

    def build(self, input_shape):
        self.filter_dense_layer = tf.keras.layers.Dense(
            self.filter_size,
            use_bias=True,
            activation=tf.nn.relu,
            name="filter_layer")
        self.output_dense_layer = tf.keras.layers.Dense(
            self.hidden_size, use_bias=True,
            name="output_layer")
        super(FeedForwardNetwork, self).build(input_shape)

    def get_config(self):
        return {
            "hidden_size": self.hidden_size,
            "filter_size": self.filter_size,
            "relu_dropout": self.relu_dropout,
        }

    def call(self, x, training):
        """Return outputs of the feedforward network.

        Args:
          x: tensor with shape [batch_size, length, hidden_size]
          training: boolean, whether in training mode or not.

        Returns:
          Output of the feedforward network.
          tensor with shape [batch_size, length, hidden_size]
        """
        # Retrieve dynamically known shapes
        batch_size = tf.shape(x)[0]  # TODO why are these two unused ?
        length = tf.shape(x)[1]

        output = self.filter_dense_layer(x)
        if training:
            output = tf.nn.dropout(output, rate=self.relu_dropout)
        output = self.output_dense_layer(output)

        return output
