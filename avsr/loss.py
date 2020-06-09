import tensorflow_addons as tfa
import tensorflow as tf
from absl import flags

FLAGS = flags.FLAGS


def make_loss(logits, label_ids, sequence_lengths=None):

    if sequence_lengths is not None:
        weights = tf.sequence_mask(
            lengths=sequence_lengths,
            maxlen=tf.shape(logits)[1],
            dtype=logits.dtype,

        )
    else:
        weights = tf.ones(tf.shape(label_ids))

    batch_loss = tfa.seq2seq.sequence_loss(
        logits=logits,
        targets=label_ids,
        weights=weights,

    )

    return batch_loss


def regularisation_loss(all_weights):
    # Todo: Use model.losses when https://github.com/tensorflow/addons/issues/577 is fixed
    weights = []
    accept_list = ['lstm', 'rnn']
    exclude_list = ['bias',]
    if FLAGS.recurrent_regularisation is False:
        exclude_list.append('recurrent_kernel')
    if FLAGS.regularise_all is False:
        exclude_list.extend(['dense', 'embedding', 'attention_g', 'memory_layer', 'attention_layer'])

    for weight in all_weights:
        name = weight.name
        if any(x in name for x in accept_list) and\
                all(x not in name for x in exclude_list):
            weights.append(weight)

    if len(weights) > 0:
        regulariser = _get_regularizer()
        loss = tf.add_n([regulariser(w) for w in weights])
    else:
        loss = 0.0
    return loss


def num_words_loss(logits, target_ids, SPACE_ID):
    num_pred_words = tf.reduce_sum(logits, axis=[1, 2], keepdims=False)
    num_true_words = tf.reduce_sum(tf.cast(tf.equal(target_ids, SPACE_ID), tf.float32), axis=1)
    word_loss = tf.losses.mean_squared_error(num_true_words, num_pred_words)

    # tf.summary.image('Num Words Logits', tf.expand_dims(tf.repeat(logits, 20, axis=-1), -1), step=1, max_outputs=3)
    # tf.summary.scalar('Num Words Loss', word_loss, step=1)

    return word_loss


def _get_regularizer():
    return tf.keras.regularizers.l1_l2(
                l1=FLAGS.rnn_l1_l2_weight_regularisation[0],
                l2=FLAGS.rnn_l1_l2_weight_regularisation[1])
