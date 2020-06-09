import tensorflow as tf
import numpy as np
import math

_NEG_INF_FP32 = -1e9
_NEG_INF_FP16 = np.finfo(np.float16).min


def get_padding(x, padding_value=0, dtype=tf.float32):
    """Return float tensor representing the padding values in x.

    Args:
      x: int tensor with any shape
      padding_value: int which represents padded values in input
      dtype: The dtype of the return value.

    Returns:
      float tensor with same shape as x containing values 0 or 1.
        0 -> non-padding, 1 -> padding
    """
    with tf.name_scope("padding"):
        return tf.cast(tf.equal(x, padding_value), dtype)


def get_padding_bias(x, padding_value=0, dtype=tf.float32):
    """Calculate bias tensor from padding values in tensor.

    Bias tensor that is added to the pre-softmax multi-headed attention logits,
    which has shape [batch_size, num_heads, length, length]. The tensor is zero at
    non-padding locations, and -1e9 (negative infinity) at padding locations.

    Args:
      x: int tensor with shape [batch_size, length]
      padding_value: int which represents padded values in input
      dtype: The dtype of the return value

    Returns:
      Attention bias tensor of shape [batch_size, 1, 1, length].
    """
    with tf.name_scope("attention_bias"):
        padding = get_padding(x, padding_value, dtype)[:, :, 0]
        attention_bias = padding * _NEG_INF_FP32
        attention_bias = tf.expand_dims(tf.expand_dims(attention_bias, axis=1), axis=1)
    return attention_bias


def get_bias_from_len(
        seq_length,
        dtype=tf.float32,
        online=False,
        lookahead=0,
        lookback=-1,
):
    with tf.name_scope("attention_bias"):
        len_mask = tf.sequence_mask(seq_length, dtype=dtype)

        if online:
            maxlen = tf.reduce_max(seq_length)
            online_mask = tf.linalg.band_part(tf.ones([maxlen, maxlen], dtype=dtype), lookback, lookahead)
            tiled_len_mask = tf.tile(tf.expand_dims(len_mask, -1), [1, 1, maxlen])
            online_len_mask = tiled_len_mask * online_mask
            attention_bias = (1.0 - online_len_mask) * _NEG_INF_FP32
            attention_bias = tf.expand_dims(attention_bias, axis=1)
        else:
            attention_bias = (1.0 - len_mask) * _NEG_INF_FP32
            attention_bias = tf.expand_dims(tf.expand_dims(attention_bias, axis=1), axis=1)

    return attention_bias


def get_bias_from_boundary_signal(
        word_sig,
        input_lengths,
        labels,
        labels_lengths,
        space_id,
        lookback=5,
        lookahead=5,
):
    max_input_length = tf.shape(word_sig)[1]
    max_output_length = tf.shape(labels)[1]

    input_cumsum = tf.math.cumsum(word_sig, axis=1)
    input_fcumsum = tf.floor(input_cumsum)

    spaces = tf.cast(labels == space_id, tf.float32)
    output_cumsum = tf.math.cumsum(spaces, axis=1)

    t1 = tf.tile(input_fcumsum, [1, 1, max_output_length])
    t2 = tf.tile(tf.expand_dims(output_cumsum, 1), [1, max_input_length, 1])

    mask = t1 <= (t2 + lookahead)
    if lookback >= 0:
        mask = tf.logical_and(mask,  t1 >= (t2 - lookback))

    # mask padded elements
    input_mask = tf.sequence_mask(input_lengths)
    output_mask = tf.sequence_mask(labels_lengths)
    tt1 = tf.tile(tf.expand_dims(input_mask, -1), [1, 1, max_output_length])
    tt2 = tf.tile(tf.expand_dims(output_mask, 1), [1, max_input_length, 1])
    len_mask = tf.logical_and(tt1, tt2)

    mask = tf.logical_and(mask, len_mask)
    mask = tf.cast(mask, tf.float32)

    bias = (1 - mask) * _NEG_INF_FP32
    bias = tf.transpose(bias, [0, 2, 1])
    bias = tf.expand_dims(bias, axis=1)

    return bias  # [bs, 1, out_len, in_len]


def get_position_encoding(
        length, hidden_size, min_timescale=1.0, max_timescale=1.0e4):
    """Return positional encoding.

    Calculates the position encoding as a mix of sine and cosine functions with
    geometrically increasing wavelengths.
    Defined and formulized in Attention is All You Need, section 3.5.

    Args:
      length: Sequence length.
      hidden_size: Size of the
      min_timescale: Minimum scale that will be applied at each position
      max_timescale: Maximum scale that will be applied at each position

    Returns:
      Tensor with shape [length, hidden_size]
    """
    # We compute the positional encoding in float32 even if the model uses
    # float16, as many of the ops used, like log and exp, are numerically unstable
    # in float16.
    position = tf.cast(tf.range(length), tf.float32)
    num_timescales = hidden_size // 2
    log_timescale_increment = (
            math.log(float(max_timescale) / float(min_timescale)) /
            (tf.cast(num_timescales, tf.float32) - 1))
    inv_timescales = min_timescale * tf.exp(
        tf.cast(tf.range(num_timescales), tf.float32) * -log_timescale_increment)
    scaled_time = tf.expand_dims(position, 1) * tf.expand_dims(inv_timescales, 0)
    signal = tf.concat([tf.sin(scaled_time), tf.cos(scaled_time)], axis=1)
    return signal


def get_decoder_self_attention_bias(length, dtype=tf.float32):
    """Calculate bias for decoder that maintains model's autoregressive property.

    Creates a tensor that masks out locations that correspond to illegal
    connections, so prediction at position i cannot draw information from future
    positions.

    Args:
      length: int length of sequences in batch.
      dtype: The dtype of the return value.

    Returns:
      float tensor of shape [1, 1, length, length]
    """
    neg_inf = _NEG_INF_FP16 if dtype == tf.float16 else _NEG_INF_FP32
    with tf.name_scope("decoder_self_attention_bias"):
        valid_locs = tf.linalg.band_part(tf.ones([length, length], dtype=dtype),
                                         -1, 0)
        valid_locs = tf.reshape(valid_locs, [1, 1, length, length])
        decoder_bias = neg_inf * (1.0 - valid_locs)
    return decoder_bias


def constrain_bias(query_length, source_length, batch_size, dtype=tf.float32, num_frames=2):
    valid_locs = tf.ones((query_length, source_length))
    ratio = tf.cast(source_length/query_length, tf.float32)

    fill_value = 0.0

    for i in tf.range(0, query_length):
        centre = (tf.cast(i, tf.float32) + 1) * ratio - 1
        nearest = tf.cast(tf.math.round(centre), tf.int32)

        # TODO are loop comprehensions supported yet ?
        indices = [[i, tf.clip_by_value(nearest + j, 0, source_length - 1)] for j in range(-num_frames, num_frames+1)]

        # loop alternative
        # indices = []
        # for j in range(-num_frames, num_frames+1):
        #     clipped_val =
        #     indices.append([i, clipped_val])

        updates = len(indices) * [fill_value]
        valid_locs = tf.tensor_scatter_nd_update(valid_locs, indices, updates)

    valid_locs = tf.expand_dims(tf.repeat(tf.expand_dims(valid_locs, 0), batch_size, axis=0), 1)

    neg_inf = _NEG_INF_FP16 if dtype == tf.float16 else _NEG_INF_FP32
    bias = neg_inf * valid_locs
    return bias


def get_online_decoder_bias(
        predicted_ids,
        space_id,
        encoder_fcumsum,
        length_mask,
        lookback=-1,
        lookahead=5,
):
    max_input_length = tf.shape(encoder_fcumsum)[1]

    spaces = tf.cast(predicted_ids == space_id, tf.float32)
    decoder_cumsum = tf.math.cumsum(spaces, axis=1)
    decoder_cumsum = decoder_cumsum[:, -1, tf.newaxis]  # because ids is a history of symbols

    decoder_cumsum = tf.tile(tf.expand_dims(decoder_cumsum, 1), [1, max_input_length, 1])

    mask = encoder_fcumsum <= (decoder_cumsum + lookahead)
    if lookback >= 0:
        mask = tf.logical_and(mask, encoder_fcumsum >= (decoder_cumsum - lookback))

    mask = tf.logical_and(mask, length_mask)
    mask = tf.cast(mask, tf.float32)

    bias = (1 - mask) * _NEG_INF_FP32
    bias = tf.transpose(bias, [0, 2, 1])
    bias = tf.expand_dims(bias, axis=1)

    return bias


def fcumsum(word_sig):
    cumsum = tf.math.cumsum(word_sig, axis=1)
    fcumsum = tf.floor(cumsum)
    return fcumsum
