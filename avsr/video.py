import tensorflow as tf
from absl import flags

FLAGS = flags.FLAGS


def build_conv2d_block(
        filters,
        kernel_size=(3, 3),
        strides=(1, 1),
        padding='same',
        activation=None,
        ):
    r"""
    Wraps Conv2D to disable bias and fix the kernel initialiser
    """
    return tf.keras.layers.Conv2D(
        filters=filters,
        kernel_size=kernel_size,
        strides=strides,
        padding=padding,
        use_bias=False,
        activation=activation,
        kernel_initializer=tf.keras.initializers.he_uniform())


class ResidualBlock(tf.keras.layers.Layer):
    def __init__(
            self,
            num_filters,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding='same',
            skip_first_norm=False,
            project_shortcut=False,
            activation=tf.nn.relu,
            norm_fn=tf.keras.layers.BatchNormalization
            ):
        r"""

        :param num_filters:
        :param kernel_size:
        :param strides:
        :param padding:
        :param activation:
        :param skip_first_norm:
        """
        super(ResidualBlock, self).__init__()
        self.skip_first_norm = skip_first_norm
        if skip_first_norm is False:
            self.norm0 = norm_fn(epsilon=1e-6, dtype="float32")
        self.norm1 = norm_fn(epsilon=1e-6, dtype="float32")

        self.project_shortcut = project_shortcut
        if project_shortcut is True:
            self.projection = build_conv2d_block(
                filters=num_filters,
                kernel_size=(1, 1),
                strides=strides,
                padding=padding,
                activation=None)

        self.conv1 = build_conv2d_block(
                filters=num_filters,
                kernel_size=kernel_size,
                strides=strides,
                padding=padding,
                activation=None)
        self.conv2 = build_conv2d_block(
                filters=num_filters,
                kernel_size=kernel_size,
                strides=(1, 1),  # second conv without stride
                padding=padding,
                activation=None)
        self.activation = activation

    def call(self, inputs, training=None, mask=None):
        r""""
        Defines the residual op
        :return:
        """
        shortcut = inputs

        if self.skip_first_norm is False:
            inputs = self.norm0(inputs, training=training)
            inputs = self.activation(inputs)

        if self.project_shortcut is True:
            shortcut = self.projection(shortcut)

        inputs = self.conv1(inputs)
        inputs = self.norm1(inputs, training=training)
        inputs = self.activation(inputs)
        inputs = self.conv2(inputs)

        output = inputs + shortcut

        return output


class VideoCNN(tf.keras.layers.Layer):
    def __init__(self,
                 cnn_filters,
                 cnn_dense_units,
                 final_activation=None):
        super(VideoCNN, self).__init__()

        self.cnn_dense_units = cnn_dense_units

        if FLAGS.cnn_activation == 'relu':
            activation = tf.nn.relu
        elif FLAGS.cnn_activation == 'elu':
            activation = tf.nn.elu
        elif FLAGS.cnn_activation == 'lrelu':
            activation = tf.nn.leaky_relu
        else:
            raise ValueError('Activation not supported')
        self.activation = activation

        self.cnn_final_clip = FLAGS.cnn_final_clip

        if FLAGS.cnn_normalisation == 'layer_norm':
            self.norm_fn = tf.keras.layers.LayerNormalization
        elif FLAGS.cnn_normalisation == 'batch_norm':
            self.norm_fn = tf.keras.layers.BatchNormalization
        else:
            raise ValueError('Unknown normalisation function')

        self.conv0 = build_conv2d_block(cnn_filters[0], activation=None)
        self.norm0 = self.norm_fn(epsilon=1e-6, dtype="float32")
        self.block0 = ResidualBlock(
            cnn_filters[0],
            project_shortcut=False,
            skip_first_norm=True,
            norm_fn=self.norm_fn,
            activation=activation)

        self.resnet_blocks = []
        for num_filters in cnn_filters:
            block = ResidualBlock(
                num_filters=num_filters,
                strides=(2, 2),
                project_shortcut=True,
                norm_fn=self.norm_fn,
                activation=activation,
            )
            self.resnet_blocks.append(block)

        self.flatten = build_conv2d_block(
            filters=self.cnn_dense_units,
            kernel_size=(3, 3),  # how to get runtime shape?
            activation=final_activation,
            padding='valid',
        )

    def call(self, inputs, training=False, mask=None):

        bs, ts, d1, d2, d3 = tf.unstack(tf.shape(inputs))
        inputs = tf.reshape(inputs, shape=tf.concat([[-1], tf.shape(inputs)[2:]], axis=0))

        flow = inputs
        flow = self.conv0(flow)
        flow = self.norm0(flow, training=training)
        flow = self.activation(flow)

        flow = self.block0(flow, training=training)

        for block in self.resnet_blocks:
            flow = block(flow, training=training)

        final = self.flatten(flow)
        if self.cnn_final_clip:
            final = tf.clip_by_value(final, -1.0, 1.0)
        final = tf.squeeze(final, axis=[1, 2])

        final = tf.reshape(final, [bs, ts, self.cnn_dense_units])

        return final
