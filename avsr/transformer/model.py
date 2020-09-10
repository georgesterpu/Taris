"""
Model originally forked from
https://github.com/tensorflow/models/tree/master/official/nlp/transformer

Main changes:
1. Adapted for speech inputs
    The audio inputs (e.g. mel spectrogram slices) are first linearly projected
    to the hidden dimension of the transformer, facilitating the first residual connections
2. Attention bias inferred from the true sequence length
    The original model assumed that all zeros must be paddings, and this is a limitation
    due to true signal zeros (e.g. coming from ReLu ops or actually raw input zeros in black images)
    or operations destroying the originally padded zeros such as Batch normalisation without masking
3. Introduced the Align stack and the AVTransformer
4. Introduced Taris for online speech recognition, consisting of:
    - a sigmoidal halting unit
    - a word loss
    - dynamic biasing of the decoder-encoder alignment weights
"""

import tensorflow as tf
from .embedding_layer import EmbeddingSharedWeights
from .utils import get_padding, get_padding_bias, get_position_encoding, get_decoder_self_attention_bias, constrain_bias
from .utils import get_bias_from_len
from .attention_layer import Attention, SelfAttention, FeedForwardNetwork
from .beam_search import sequence_beam_search
from absl import flags
from ..video import VideoCNN
from ..loss import num_words_loss
from .utils import get_bias_from_boundary_signal, get_online_decoder_bias, fcumsum

FLAGS = flags.FLAGS


class Transformer(tf.keras.Model):
    """Transformer model with Keras.

    Implemented as described in: https://arxiv.org/pdf/1706.03762.pdf

    The Transformer model consists of an encoder and decoder. The input is an int
    sequence (or a batch of sequences). The encoder produces a continuous
    representation, and the decoder uses the encoder output to generate
    probabilities for the output sequence.
    """

    def __init__(self, unit_dict, name=None):
        """Initialize layers to build Transformer model.
        Args:
          name: name of the model.
        """
        super(Transformer, self).__init__(name=name)
        self.unit_dict = unit_dict
        self.reverse_dict = {v: k for k, v in self.unit_dict.items()}
        self.vocab_size = len(self.unit_dict) - 1
        self.SPACE_ID = self.reverse_dict[' ']

        self.use_word_loss = FLAGS.word_loss

        self.embedding_softmax_layer = EmbeddingSharedWeights(
            self.vocab_size, FLAGS.transformer_hidden_size)

        if FLAGS.architecture == 'transformer':
            # is it worth it to specialise this class for single input modalities ?
            self.encoder_stack = EncoderStack()
            self.decoder_stack = DecoderStack()

        if FLAGS.wb_activation == 'sigmoid':
            self.wb_activation = tf.math.sigmoid
        elif FLAGS.wb_activation == 'tanh':
            self.wb_activation = tf.math.tanh
        else:
            self.wb_activation = tf.math.sigmoid

    def build(self, input_shape):
        self.input_dense_layer = tf.keras.layers.Dense(
            units=FLAGS.transformer_hidden_size,
            kernel_regularizer=tf.keras.regularizers.l1_l2(
                l1=FLAGS.transformer_l1_regularisation,
                l2=FLAGS.transformer_l2_regularisation),
            )
        if self.use_word_loss or FLAGS.transformer_online_decoder:
            self.gate = tf.keras.layers.Dense(
                units=1,
                kernel_regularizer=tf.keras.regularizers.l1_l2(
                    l1=FLAGS.transformer_l1_regularisation,
                    l2=FLAGS.transformer_l2_regularisation),
                activation=self.wb_activation)

    def call(self, inputs, training):
        """Calculate target logits or inferred target sequences.

        Args:
          inputs: input tensor list of size 2.
            First item, inputs: int tensor with shape [batch_size, input_length].
            Second item (optional), targets: int tensor with shape
              [batch_size, target_length].
          training: boolean, whether in training mode or not.

        Returns:
          If targets is defined, then return logits for each word in the target
          sequence. float tensor with shape [batch_size, target_length, vocab_size]
          If target is none, then generate output sequence one token at a time.
            returns a dictionary {
              outputs: [batch_size, decoded length]
              scores: [batch_size, float]}
          Even when float16 is used, the output tensor(s) are always float32.
        """
        encoder_inputs, targets = inputs.inputs, inputs.labels
        encoder_inputs_length = inputs.inputs_length
        decoder_inputs_length = inputs.labels_length

        # Variance scaling is used here because it seems to work in many problems.
        # Other reasonable initializers may also work just as well.
        with tf.name_scope("Transformer"):
            # Step 0 for ASR inputs: linear projection to hidden_size
            encoder_inputs = self.input_dense_layer(encoder_inputs)

            # Calculate attention bias for encoder self-attention and decoder
            # multi-headed attention layers.
            attention_bias = get_bias_from_len(
                encoder_inputs_length,
                online=FLAGS.transformer_online_encoder,
                lookahead=FLAGS.transformer_encoder_lookahead,
                lookback=FLAGS.transformer_encoder_lookback)

            # Run the inputs through the encoder layer to map the symbol
            # representations to continuous representations.
            encoder_outputs = self.encode(encoder_inputs, self.encoder_stack, attention_bias, training)
            # Generate output sequence if targets is None, or return logits if target
            # sequence is known.

            if self.use_word_loss or FLAGS.transformer_online_decoder:
                word_sig = self.gate(encoder_outputs)

                word_loss = num_words_loss(
                    word_sig,
                    targets,
                    self.SPACE_ID
                )
                self.add_loss(FLAGS.wordloss_weight * word_loss)

                word_fcumsum = fcumsum(word_sig)
                extra = {'word_logits': word_sig,
                         'word_floor_cumsum': word_fcumsum}
            else:
                extra = {}

            if training:
                if FLAGS.transformer_online_decoder:
                    attention_bias = get_bias_from_boundary_signal(
                        word_sig,
                        encoder_inputs_length,
                        targets,
                        decoder_inputs_length,
                        self.SPACE_ID,
                        lookahead=FLAGS.transformer_decoder_lookahead,
                        lookback=FLAGS.transformer_decoder_lookback,
                    )
                else:
                    attention_bias = get_bias_from_len(encoder_inputs_length)

                logits = self.decode(targets, encoder_outputs, attention_bias, training)
                return logits, extra
            else:
                if FLAGS.transformer_online_decoder:
                    bias = None
                else:
                    word_fcumsum = None
                    bias = get_bias_from_len(encoder_inputs_length)

                num_target_words = tf.math.cumsum(tf.cast(tf.equal(targets, self.SPACE_ID), tf.float32), axis=1)[:,-1]
                output = self.predict(
                    encoder_outputs, encoder_inputs_length, training,
                    encoder_decoder_attention_bias=bias,
                    word_fcumsum=word_fcumsum,
                    target_words=num_target_words)
                return output, extra

    def encode(self, inputs, encoder_stack, attention_bias, training):
        """Generate continuous representation for inputs.

        Args:
          inputs: int tensor with shape [batch_size, input_length].
          attention_bias: float tensor with shape [batch_size, 1, 1, input_length].
          training: boolean, whether in training mode or not.

        Returns:
          float tensor with shape [batch_size, input_length, hidden_size]
        """
        with tf.name_scope("encode"):
            # Prepare inputs to the layer stack by adding positional encodings and
            # applying dropout.

            # embedded_inputs = self.embedding_softmax_layer(inputs)
            embedded_inputs = inputs
            embedded_inputs = tf.cast(embedded_inputs, FLAGS.transformer_dtype)

            inputs_padding = get_padding(inputs)
            attention_bias = tf.cast(attention_bias, FLAGS.transformer_dtype)

            with tf.name_scope("add_pos_encoding"):
                _, input_length, input_size = tf.unstack(tf.shape(embedded_inputs))
                pos_encoding = get_position_encoding(
                    input_length, input_size)
                pos_encoding = tf.cast(pos_encoding, FLAGS.transformer_dtype)
                encoder_inputs = embedded_inputs + pos_encoding

            if training:
                encoder_inputs = tf.nn.dropout(
                    encoder_inputs, rate=FLAGS.transformer_layer_postprocess_dropout)

            return encoder_stack(
                encoder_inputs, attention_bias, inputs_padding, training=training)

    def decode(self, targets, encoder_outputs, attention_bias, training):
        """Generate logits for each value in the target sequence.

        Args:
          targets: target values for the output sequence. int tensor with shape
            [batch_size, target_length]
          encoder_outputs: continuous representation of input sequence. float tensor
            with shape [batch_size, input_length, hidden_size]
          attention_bias: float tensor with shape [batch_size, 1, 1, input_length]
          training: boolean, whether in training mode or not.

        Returns:
          float32 tensor with shape [batch_size, target_length, vocab_size]
        """
        with tf.name_scope("decode"):
            # Prepare inputs to decoder layers by shifting targets, adding positional
            # encoding and applying dropout.
            decoder_inputs = self.embedding_softmax_layer(targets)
            decoder_inputs = tf.cast(decoder_inputs, FLAGS.transformer_dtype)
            attention_bias = tf.cast(attention_bias, FLAGS.transformer_dtype)
            with tf.name_scope("shift_targets"):
                # Shift targets to the right, and remove the last element
                decoder_inputs = tf.pad(decoder_inputs,
                                        [[0, 0], [1, 0], [0, 0]])[:, :-1, :]
            with tf.name_scope("add_pos_encoding"):
                length = tf.shape(decoder_inputs)[1]
                pos_encoding = get_position_encoding(
                    length, FLAGS.transformer_hidden_size)
                pos_encoding = tf.cast(pos_encoding, FLAGS.transformer_dtype)
                decoder_inputs += pos_encoding
            if training:
                decoder_inputs = tf.nn.dropout(
                    decoder_inputs, rate=FLAGS.transformer_layer_postprocess_dropout)

            # Run values
            decoder_self_attention_bias = get_decoder_self_attention_bias(
                length, dtype=FLAGS.transformer_dtype)
            outputs = self.decoder_stack(
                decoder_inputs,
                encoder_outputs,
                decoder_self_attention_bias,
                attention_bias,
                training=training)
            logits = self.embedding_softmax_layer(outputs, mode="linear")
            logits = tf.cast(logits, tf.float32)
            return logits

    def _get_symbols_to_logits_fn(self, max_decode_length, training):
        """Returns a decoding function that calculates logits of the next tokens."""

        timing_signal = get_position_encoding(
            max_decode_length + 1, FLAGS.transformer_hidden_size)
        timing_signal = tf.cast(timing_signal, FLAGS.transformer_dtype)
        decoder_self_attention_bias = get_decoder_self_attention_bias(
            max_decode_length, dtype=FLAGS.transformer_dtype)

        # TODO(b/139770046): Refactor code with better naming of i.
        def symbols_to_logits_fn(ids, i, cache):
            """Generate logits for next potential IDs.

            Args:
              ids: Current decoded sequences. int tensor with shape [batch_size *
                beam_size, i + 1].
              i: Loop index.
              cache: dictionary of values storing the encoder output, encoder-decoder
                attention bias, and previous decoder attention values.

            Returns:
              Tuple of
                (logits with shape [batch_size * beam_size, vocab_size],
                 updated cache values)
            """
            # Set decoder input to the last generated IDs
            decoder_input = ids[:, -1:]

            # Preprocess decoder input by getting embeddings and adding timing signal.
            decoder_input = self.embedding_softmax_layer(decoder_input)

            decoder_input += timing_signal[i:i + 1]

            self_attention_bias = decoder_self_attention_bias[:, :, i:i + 1, :i + 1]

            if FLAGS.transformer_online_decoder:
                timestep_bias = get_online_decoder_bias(
                    ids,
                    self.SPACE_ID,
                    encoder_fcumsum=cache.get("word_fcumsum"),
                    length_mask=cache.get("input_mask"),
                    lookahead=FLAGS.transformer_decoder_lookahead,
                    lookback=FLAGS.transformer_decoder_lookback,
                )
            else:
                timestep_bias = cache.get("encoder_decoder_attention_bias")

            decoder_outputs = self.decoder_stack(
                decoder_input,
                cache.get("encoder_outputs"),
                self_attention_bias,
                timestep_bias,
                training=training,
                cache=cache)
            logits = self.embedding_softmax_layer(decoder_outputs, mode="linear")
            logits = tf.squeeze(logits, axis=[1])
            return logits, cache

        return symbols_to_logits_fn

    def predict(self, encoder_outputs, encoder_output_length, training,
                target_words,
                encoder_decoder_attention_bias=None,
                word_fcumsum=None,
                ):
        """Return predicted sequence."""
        encoder_outputs = tf.cast(encoder_outputs, FLAGS.transformer_dtype)
        batch_size = tf.shape(encoder_outputs)[0]
        input_length = tf.shape(encoder_outputs)[1]
        max_decode_length = input_length + FLAGS.transformer_extra_decode_length

        symbols_to_logits_fn = self._get_symbols_to_logits_fn(
            max_decode_length, training)

        # Create initial set of IDs that will be passed into symbols_to_logits_fn.
        initial_ids = tf.zeros([batch_size], dtype=tf.int32)

        # Create cache storing decoder attention values for each layer.
        # pylint: disable=g-complex-comprehension
        init_decode_length = 0
        num_heads = FLAGS.transformer_num_heads
        dim_per_head = FLAGS.transformer_hidden_size // num_heads
        cache = {
            "layer_%d" % layer: {
                "k":
                    tf.zeros([
                        batch_size, init_decode_length, num_heads, dim_per_head
                    ],
                        dtype=FLAGS.transformer_dtype),
                "v":
                    tf.zeros([
                        batch_size, init_decode_length, num_heads, dim_per_head
                    ],
                        dtype=FLAGS.transformer_dtype)
            } for layer in range(FLAGS.transformer_num_decoder_layers)
        }
        # pylint: enable=g-complex-comprehension

        # Add encoder output and attention bias to the cache.
        cache["encoder_outputs"] = encoder_outputs
        cache["target_words"] = target_words

        if FLAGS.transformer_online_decoder:
            cache["input_mask"] = tf.expand_dims(tf.sequence_mask(encoder_output_length), -1)
            cache["word_fcumsum"] = word_fcumsum
        else:
            cache["encoder_decoder_attention_bias"] = tf.cast(encoder_decoder_attention_bias, FLAGS.transformer_dtype)

        # Use beam search to find the top beam_size sequences and scores.
        decoded_ids, scores = sequence_beam_search(
            symbols_to_logits_fn=symbols_to_logits_fn,
            initial_ids=initial_ids,
            initial_cache=cache,
            vocab_size=self.vocab_size,
            beam_size=FLAGS.transformer_beam_size,
            alpha=FLAGS.transformer_alpha,
            max_decode_length=max_decode_length,
            eos_id=self.reverse_dict[' '],
            dtype=FLAGS.transformer_dtype)

        # Get the top sequence for each batch element
        top_decoded_ids = decoded_ids[:, 0, 1:]
        top_scores = scores[:, 0]

        return {"outputs": top_decoded_ids, "scores": top_scores}


class PrePostProcessingWrapper(tf.keras.layers.Layer):
    """Wrapper class that applies layer pre-processing and post-processing."""

    def __init__(self, layer):
        super(PrePostProcessingWrapper, self).__init__()
        self.layer = layer
        self.postprocess_dropout = FLAGS.transformer_layer_postprocess_dropout

    def build(self, input_shape):
        # Create normalization layer
        self.layer_norm = tf.keras.layers.LayerNormalization(
            epsilon=1e-6, dtype="float32")
        super(PrePostProcessingWrapper, self).build(input_shape)

    def call(self, x, *args, **kwargs):
        """Calls wrapped layer with same parameters."""
        # Preprocessing: apply layer normalization
        training = kwargs["training"]

        y = self.layer_norm(x)

        # Get layer output
        y = self.layer(y, *args, **kwargs)

        # Postprocessing: apply dropout and residual connection
        if training:
            y = tf.nn.dropout(y, rate=self.postprocess_dropout)
        return x + y


class EncoderStack(tf.keras.layers.Layer):
    """Transformer encoder stack.

    The encoder stack is made up of N identical layers. Each layer is composed
    of the sublayers:
      1. Self-attention layer
      2. Feedforward network (which is 2 fully-connected layers)
    """

    def __init__(self):
        super(EncoderStack, self).__init__()
        self.layers = []

    def build(self, input_shape):
        """Builds the encoder stack."""
        for _ in range(FLAGS.transformer_num_encoder_layers):
            # Create sublayers for each layer.
            self_attention_layer = SelfAttention(
                FLAGS.transformer_hidden_size, FLAGS.transformer_num_heads,
                FLAGS.transformer_attention_dropout)
            feed_forward_network = FeedForwardNetwork(
                FLAGS.transformer_hidden_size, FLAGS.transformer_filter_size, FLAGS.transformer_relu_dropout)

            self.layers.append([
                PrePostProcessingWrapper(self_attention_layer),
                PrePostProcessingWrapper(feed_forward_network)
            ])

        # Create final layer normalization layer.
        self.output_normalization = tf.keras.layers.LayerNormalization(
            epsilon=1e-6, dtype="float32")
        super(EncoderStack, self).build(input_shape)

    def call(self, encoder_inputs, attention_bias, inputs_padding, training):
        """Return the output of the encoder layer stacks.

        Args:
          encoder_inputs: tensor with shape [batch_size, input_length, hidden_size]
          attention_bias: bias for the encoder self-attention layer. [batch_size, 1, 1, input_length]
          inputs_padding: tensor with shape [batch_size, input_length], inputs with zero paddings.
          training: boolean, whether in training mode or not.

        Returns:
          Output of encoder layer stack.
          float32 tensor with shape [batch_size, input_length, hidden_size]
        """
        for n, layer in enumerate(self.layers):
            # Run inputs through the sublayers.
            self_attention_layer = layer[0]
            feed_forward_network = layer[1]

            with tf.name_scope("layer_%d" % n):
                with tf.name_scope("self_attention"):
                    encoder_inputs = self_attention_layer(
                        encoder_inputs, attention_bias, training=training)
                with tf.name_scope("ffn"):
                    encoder_inputs = feed_forward_network(
                        encoder_inputs, training=training)

        return self.output_normalization(encoder_inputs)


class DecoderStack(tf.keras.layers.Layer):
    """Transformer decoder stack.

    Like the encoder stack, the decoder stack is made up of N identical layers.
    Each layer is composed of the sublayers:
      1. Self-attention layer
      2. Multi-headed attention layer combining encoder outputs with results from
         the previous self-attention layer.
      3. Feedforward network (2 fully-connected layers)
    """

    def __init__(self):
        super(DecoderStack, self).__init__()
        self.layers = []

    def build(self, input_shape):
        """Builds the decoder stack."""
        for _ in range(FLAGS.transformer_num_decoder_layers):
            self_attention_layer = SelfAttention(
                FLAGS.transformer_hidden_size,FLAGS.transformer_num_heads,
                FLAGS.transformer_attention_dropout)
            enc_dec_attention_layer = Attention(
                FLAGS.transformer_hidden_size, FLAGS.transformer_num_heads,
                FLAGS.transformer_attention_dropout)
            feed_forward_network = FeedForwardNetwork(
                FLAGS.transformer_hidden_size, FLAGS.transformer_filter_size, FLAGS.transformer_relu_dropout)

            self.layers.append([
                PrePostProcessingWrapper(self_attention_layer),
                PrePostProcessingWrapper(enc_dec_attention_layer),
                PrePostProcessingWrapper(feed_forward_network)
            ])
        self.output_normalization = tf.keras.layers.LayerNormalization(
            epsilon=1e-6, dtype="float32")
        super(DecoderStack, self).build(input_shape)

    def call(self,
             decoder_inputs,
             encoder_outputs,
             decoder_self_attention_bias,
             attention_bias,
             training,
             cache=None):
        """Return the output of the decoder layer stacks.

        Args:
          decoder_inputs: A tensor with shape
            [batch_size, target_length, hidden_size].
          encoder_outputs: A tensor with shape
            [batch_size, input_length, hidden_size]
          decoder_self_attention_bias: A tensor with shape
            [1, 1, target_len, target_length], the bias for decoder self-attention
            layer.
          attention_bias: A tensor with shape [batch_size, 1, 1, input_length],
            the bias for encoder-decoder attention layer.
          training: A bool, whether in training mode or not.
          cache: (Used for fast decoding) A nested dictionary storing previous
            decoder self-attention values. The items are:
              {layer_n: {"k": A tensor with shape [batch_size, i, key_channels],
                         "v": A tensor with shape [batch_size, i, value_channels]},
                           ...}
        Returns:
          Output of decoder layer stack.
          float32 tensor with shape [batch_size, target_length, hidden_size]
        """
        for n, layer in enumerate(self.layers):
            self_attention_layer = layer[0]
            enc_dec_attention_layer = layer[1]
            feed_forward_network = layer[2]

            # Run inputs through the sublayers.
            layer_name = "layer_%d" % n
            layer_cache = cache[layer_name] if cache is not None else None
            with tf.name_scope(layer_name):
                with tf.name_scope("self_attention"):
                    decoder_inputs = self_attention_layer(
                        decoder_inputs,
                        decoder_self_attention_bias,
                        training=training,
                        cache=layer_cache)
                with tf.name_scope("encdec_attention"):
                    decoder_inputs = enc_dec_attention_layer(
                        decoder_inputs,
                        encoder_outputs,
                        attention_bias,
                        training=training)
                with tf.name_scope("ffn"):
                    decoder_inputs = feed_forward_network(
                        decoder_inputs, training=training)

        return self.output_normalization(decoder_inputs)


class AlignStack(tf.keras.layers.Layer):
    def __init__(self):
        super(AlignStack, self).__init__()
        self.layers = []
        self.constrain_attention = FLAGS.constrain_av_attention

    def build(self, input_shape):
        """Builds the AV Align stack."""
        for _ in range(FLAGS.transformer_num_avalign_layers):
            audio_video_attention_layer = Attention(
                FLAGS.transformer_hidden_size,
                FLAGS.transformer_num_heads,
                FLAGS.transformer_attention_dropout)
            ffn_layer = FeedForwardNetwork(
                FLAGS.transformer_hidden_size,
                FLAGS.transformer_filter_size,
                FLAGS.transformer_relu_dropout)

            self.layers.append([
                PrePostProcessingWrapper(audio_video_attention_layer),
                PrePostProcessingWrapper(ffn_layer)
            ])
        self.output_normalization = tf.keras.layers.LayerNormalization(
            epsilon=1e-6, dtype="float32")
        super(AlignStack, self).build(input_shape)

    def call(self, audio_memory, video_memory, attention_bias, training):

        if self.constrain_attention is True:
            constrained_bias = constrain_bias(
                query_length=tf.shape(audio_memory)[1],
                source_length=tf.shape(video_memory)[1],
                batch_size=tf.shape(audio_memory)[0],
                num_frames=FLAGS.constrain_frames)

            attention_bias += constrained_bias

        for n, layer in enumerate(self.layers):
            av_layer = layer[0]
            ffn_layer = layer[1]
            layer_name = "layer_%d" % n

            with tf.name_scope(layer_name):
                with tf.name_scope("AV_attention"):
                    audio_memory = av_layer(
                        audio_memory,
                        video_memory,
                        attention_bias,
                        training=training
                    )
                with tf.name_scope("ffn"):
                    audio_memory = ffn_layer(audio_memory, training=training)

        return self.output_normalization(audio_memory)


class AVTransformer(Transformer):
    def __init__(self, unit_dict, name=None):
        """Initialize layers to build Transformer model.
        Args:
          name: name of the model.
        """
        super(AVTransformer, self).__init__(unit_dict=unit_dict, name=name)

        self.use_au_loss = FLAGS.au_loss

        # The super class won't instantiate its encode/decode stacks
        # but maybe there is a better design without overdoing abstractions
        # e.g. can these be moved to build() ?
        self.audio_encoder_stack = EncoderStack()
        self.video_encoder_stack = EncoderStack()
        self.video_cnn = VideoCNN(
            cnn_filters=FLAGS.cnn_filters,
            cnn_dense_units=FLAGS.cnn_dense_units,
            final_activation=FLAGS.cnn_final_activation)
        self.align_stack = AlignStack()
        self.decoder_stack = DecoderStack()

    def build(self, input_shape):
        self.input_dense_layer = tf.keras.layers.Dense(units=FLAGS.transformer_hidden_size)
        if self.use_au_loss:
            self.au_layer = tf.keras.layers.Dense(units=2, activation='sigmoid')
            self.mse = tf.keras.losses.MeanSquaredError()
        if self.use_word_loss or FLAGS.transformer_online_decoder:
            self.gate = tf.keras.layers.Dense(1, activation='sigmoid')

    def call(self, inputs, training):
        """Calculate target logits or inferred target sequences.

        Args:
          inputs: input tensor list of size 2.
            First item, inputs: int tensor with shape [batch_size, input_length].
            Second item (optional), targets: int tensor with shape
              [batch_size, target_length].
          training: boolean, whether in training mode or not.

        Returns:
          If targets is defined, then return logits for each word in the target
          sequence. float tensor with shape [batch_size, target_length, vocab_size]
          If target is none, then generate output sequence one token at a time.
            returns a dictionary {
              outputs: [batch_size, decoded length]
              scores: [batch_size, float]}
          Even when float16 is used, the output tensor(s) are always float32.
        """
        video_input, audio_input = inputs.inputs
        video_len, audio_len = inputs.inputs_length
        targets, targets_length = inputs.labels, inputs.labels_length

        with tf.name_scope("AVTransformer"):

            # Step 1 : encode Video frames
            encoded_video_frames = self.video_cnn(video_input, training=training)
            video_attention_bias = get_bias_from_len(
                video_len,
                online=FLAGS.transformer_online_encoder,
                lookahead=FLAGS.transformer_encoder_lookahead,
                lookback=FLAGS.transformer_encoder_lookback)

            video_encoder_outputs = self.encode(
                inputs=encoded_video_frames,
                encoder_stack=self.video_encoder_stack,
                attention_bias=video_attention_bias,
                training=training)

            if self.use_au_loss is True:
                aus = self.au_layer(video_encoder_outputs)
                aus_labels = tf.clip_by_value(inputs.payload['aus'], 0.0, 3.0) / 3.0

                mask = tf.sequence_mask(inputs.inputs_length[0])
                # mask = tf.expand_dims(mask, -1)

                au_loss = self.mse(
                    y_true=aus_labels,
                    y_pred=aus,
                    sample_weight=mask)
                au_loss = tf.cast(au_loss, tf.float32)
                tf.summary.scalar('Action Unit Loss', au_loss, step=1)
                self.add_loss(FLAGS.au_loss_weight * au_loss)

            # Step 2: encode Audio features
            audio_encoder_inputs = self.input_dense_layer(audio_input)

            audio_attention_bias = get_bias_from_len(
                audio_len,
                online=FLAGS.transformer_online_encoder,
                lookahead=FLAGS.transformer_encoder_lookahead,
                lookback=FLAGS.transformer_encoder_lookback)

            audio_encoder_outputs = self.encode(
                inputs=audio_encoder_inputs,
                encoder_stack=self.audio_encoder_stack,
                attention_bias=audio_attention_bias,
                training=training)

            # Step 3: align and fuse
            #       video_encoder_outputs with
            #       audio_encoder_outputs

            fused_encoder_outputs = self.align_stack(
                audio_encoder_outputs,
                video_memory=video_encoder_outputs,
                attention_bias=video_attention_bias,
                # attention_bias=get_bias_from_len(video_len),
                training=training)

            if self.use_word_loss or FLAGS.transformer_online_decoder:
                word_sig = self.gate(fused_encoder_outputs)

                word_loss = num_words_loss(
                    word_sig,
                    targets,
                    self.SPACE_ID
                )
                self.add_loss(FLAGS.wordloss_weight * word_loss)

                word_fcumsum = fcumsum(word_sig)
                extra = {'word_logits': word_sig,
                         'word_floor_cumsum': word_fcumsum}
            else:
                extra = {}

            if training:
                if FLAGS.transformer_online_decoder:
                    attention_bias = get_bias_from_boundary_signal(
                        word_sig,
                        audio_len,
                        targets,
                        targets_length,
                        self.SPACE_ID,
                        lookahead=FLAGS.transformer_decoder_lookahead,
                        lookback=FLAGS.transformer_decoder_lookback,
                    )
                else:
                    attention_bias = get_bias_from_len(audio_len)

                logits = self.decode(targets, fused_encoder_outputs, attention_bias, training)
                return logits, extra
            else:
                if FLAGS.transformer_online_decoder:
                    bias = None
                else:
                    bias = get_bias_from_len(audio_len)
                    word_fcumsum = None

                num_target_words = tf.math.cumsum(tf.cast(tf.equal(targets, self.SPACE_ID), tf.float32), axis=1)[:, -1]
                output = self.predict(
                    fused_encoder_outputs, audio_len, training,
                    encoder_decoder_attention_bias=bias,
                    word_fcumsum=word_fcumsum,
                    target_words=num_target_words)
                return output, extra
