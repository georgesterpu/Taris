import tensorflow as tf
import collections
from absl import flags
from os import path
from random import shuffle

FLAGS = flags.FLAGS


BatchedData = collections.namedtuple(
    "BatchedData",
    (
        "inputs",
        "inputs_length",
        "inputs_filenames",
        "labels",
        "labels_length",
        "labels_filenames",
        "payload",
    ))


def _get_input_shape_from_record(record):
    dataset = tf.data.TFRecordDataset(record)
    it = iter(dataset)
    elem = next(it)

    example = tf.train.SequenceExample()
    example.ParseFromString(elem.numpy())

    content_type = {}

    context = dict(example.context.feature)
    if "input_size" in context.keys():
        content_type['stream'] = 'feature'
        input_shape = context["input_size"].int64_list.value[0]
        input_shape = [input_shape]
    elif "width" in context.keys():
        width = context["width"].int64_list.value[0]
        height = context["height"].int64_list.value[0]

        if "channels" in context.keys():
            channels = context["channels"].int64_list.value[0]
        else:
            channels = 1

        content_type['stream'] = 'video'
        input_shape = [width, height, channels]

        feature_list = dict(example.feature_lists.feature_list)
        if 'aus' in feature_list:
            content_type['aus'] = True
    else:
        raise Exception('the record is neither a video nor a feature stream)')

    del elem, it, dataset, example  # is it necessary ?
    return input_shape, content_type


def _parse_input_function(example, input_shape, content_type):

    if content_type.get('stream', False) == 'feature':
        context_features = {
            "input_length": tf.io.FixedLenFeature(shape=[], dtype=tf.int64),
            "input_size": tf.io.FixedLenFeature(shape=[], dtype=tf.int64),
            "filename": tf.io.FixedLenFeature(shape=[], dtype=tf.string)
        }
    elif content_type.get('stream', None) == 'video':
        context_features = {
            "input_length": tf.io.FixedLenFeature(shape=[], dtype=tf.int64),
            "width": tf.io.FixedLenFeature(shape=[], dtype=tf.int64),
            "height": tf.io.FixedLenFeature(shape=[], dtype=tf.int64),
            "channels": tf.io.FixedLenFeature(shape=[], dtype=tf.int64),
            "filename": tf.io.FixedLenFeature(shape=[], dtype=tf.string)
        }
    else:
        raise Exception('unknown content type')

    sequence_features = {
        "inputs": tf.io.FixedLenSequenceFeature(shape=input_shape, dtype=tf.float32)
    }

    if content_type.get('aus', False):
        sequence_features['aus'] = tf.io.FixedLenSequenceFeature(shape=[2], dtype=tf.float32)

    context_parsed, sequence_parsed = tf.io.parse_single_sequence_example(
        serialized=example,
        context_features=context_features,
        sequence_features=sequence_features
    )

    sequence_output = [sequence_parsed["inputs"], ]

    if content_type.get('aus', False):
        sequence_output.append(sequence_parsed["aus"])

    context_output = [context_parsed["input_length"], context_parsed["filename"]]

    return sequence_output + context_output


def _parse_labels_function(example, unit_dict):
    context_features = {
        "unit": tf.io.FixedLenFeature(shape=[], dtype=tf.string),
        "labels_length": tf.io.FixedLenFeature(shape=[], dtype=tf.int64),
        "filename": tf.io.FixedLenFeature(shape=[], dtype=tf.string)
    }
    sequence_features = {
        "labels": tf.io.FixedLenSequenceFeature(shape=[], dtype=tf.int64)
    }

    context_parsed, sequence_parsed = tf.io.parse_single_sequence_example(
        serialized=example,
        context_features=context_features,
        sequence_features=sequence_features
    )

    ivdict = {v: k for k, v in unit_dict.items()}
    labels = tf.concat([sequence_parsed["labels"], [ivdict[' ']]], axis=0)

    labels_length = context_parsed["labels_length"] + 1

    return labels, labels_length, context_parsed["filename"], context_parsed["unit"]


def create_unit_dict(unit_file):

    unit_dict = {'MASK': 0, 'END': -1}

    with open(unit_file, 'r') as f:
        unit_list = f.read().splitlines()

    idx = 0
    for idx, subunit in enumerate(unit_list):
        unit_dict[subunit] = idx + 1

    # unit_dict['EOS'] = idx + 2
    unit_dict['GO'] = idx + 2

    ivdict = {v: k for k, v in unit_dict.items()}

    return ivdict


def make_stream_dataset(record, num_cores=None):
    dataset = tf.data.TFRecordDataset(record, num_parallel_reads=num_cores)
    input_shape, content_type = _get_input_shape_from_record(record)

    dataset = dataset.map(
        lambda example: _parse_input_function(
            example,
            input_shape=input_shape,
            content_type=content_type),
        num_parallel_calls=num_cores)

    return dataset, input_shape, content_type


def make_labels_dataset(record, unit_dict, num_cores=None):
    dataset = tf.data.TFRecordDataset(record, num_parallel_reads=num_cores)
    dataset = dataset.map(lambda example: _parse_labels_function(example, unit_dict), num_parallel_calls=num_cores)
    return dataset


def make_unimodal_dataset(
        data_record,
        label_record,
        unit_list_file,
        shuffle=False,
        batch_size=32,
        bucket_width=-1,
        num_cores=4):

    features, features_shape, content_type = make_stream_dataset(data_record, num_cores=num_cores)
    labels = make_labels_dataset(label_record, unit_dict=create_unit_dict(unit_list_file), num_cores=num_cores)

    has_aus = content_type.get('aus', False)

    dataset = tf.data.Dataset.zip((features, labels))

    if shuffle is True:
        dataset = dataset.shuffle(buffer_size=FLAGS.buffer_size, reshuffle_each_iteration=True)

    def batching_fun(x):
        if has_aus:
            input_shape = (tf.TensorShape([None] + features_shape), tf.TensorShape([None, 2]), tf.TensorShape([]),
                           tf.TensorShape([]))
        else:
            input_shape = (tf.TensorShape([None] + features_shape), tf.TensorShape([]), tf.TensorShape([]))

        labels_shape = (tf.TensorShape([None]), tf.TensorShape([]), tf.TensorShape([]), tf.TensorShape([]))
        return x.padded_batch(
            batch_size=batch_size,
            padded_shapes=(
                input_shape,
                labels_shape
            ), drop_remainder=False,
        )

    if bucket_width == -1:
        dataset = batching_fun(dataset)
    else:
        def key_func(data, _labels):
            # inputs_len = tf.shape(arg1[0])[0]
            bucket_id = data[-2] // bucket_width
            # return tf.cast(bucket_id, dtype=tf.int64)
            return bucket_id

        def reduce_func(_unused_key, windowed_dataset):
            return batching_fun(windowed_dataset)

        dataset = tf.data.Dataset.apply(dataset, tf.data.experimental.group_by_window(
            key_func=key_func, reduce_func=reduce_func, window_size=batch_size))

    # dataset = dataset.map(lambda x, y: _make_namedtuple(x, y))
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    info = {'has_aus': has_aus}
    return dataset, info


def make_bimodal_dataset(
        video_record,
        audio_record,
        label_record,
        unit_list_file,
        shuffle=False,
        batch_size=32,
        bucket_width=-1,
        num_cores=4):

    video, video_input_shape, vid_content_type = make_stream_dataset(video_record)
    audio, audio_input_shape, _ = make_stream_dataset(audio_record)
    labels = make_labels_dataset(label_record, unit_dict=create_unit_dict(unit_list_file), num_cores=num_cores)

    has_aus = vid_content_type.get('aus', False)

    dataset = tf.data.Dataset.zip((video, audio, labels))

    if shuffle is True:
        dataset = dataset.shuffle(buffer_size=FLAGS.buffer_size, reshuffle_each_iteration=True)

    def batching_fun(x, _has_aus):

        if _has_aus:
            video_shape = (
                tf.TensorShape([None] + video_input_shape), tf.TensorShape([None, 2]), tf.TensorShape([]),
                tf.TensorShape([]),)
        else:
            video_shape = (
                tf.TensorShape([None] + video_input_shape), tf.TensorShape([]), tf.TensorShape([]),)

        audio_shape = (tf.TensorShape([None] + audio_input_shape), tf.TensorShape([]), tf.TensorShape([]),)
        labels_shape = (tf.TensorShape([None]), tf.TensorShape([]), tf.TensorShape([]), tf.TensorShape([]))

        return x.padded_batch(
            batch_size=batch_size,
            padded_shapes=(video_shape, audio_shape, labels_shape)
        )

    if bucket_width == -1:
        dataset = batching_fun(dataset, has_aus)
    else:

        def key_func(_vid, _aud, _labels):
            # video length: _vid[-1]
            # audio length: _aud[1]
            # labels length: _labels[1]
            bucket_id = _aud[1] // bucket_width
            return bucket_id

        def reduce_func(_unused_key, windowed_dataset):
            return batching_fun(windowed_dataset, has_aus)

        dataset = tf.data.Dataset.apply(dataset, tf.data.experimental.group_by_window(
            key_func=key_func, reduce_func=reduce_func, window_size=batch_size))

    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    info = {'has_aus': has_aus}
    return dataset, info


def structure_data(batch, num_streams=1, info=None):

    payload = {}
    has_aus = info.get('has_aus', False)
    if has_aus:
        payload['aus'] = batch[0][1]

    if num_streams == 1:
        float_data = batch[0][0]
        # if FLAGS.transformer_dtype == 'float16':
        #     float_data = tf.cast(float_data, tf.float16)

        # if FLAGS.mix_noise is True:
        #     import tensorflow_probability as tfp
        #     if FLAGS.noise_level == 3:
        #         probs = [0.25, 0.25, 0.25, 0.25]
        #     elif FLAGS.noise_level == 2:
        #         probs = [0.33, 0.33, 0.33, 0.0]
        #     elif FLAGS.noise_level == 1:
        #         probs = [0.5, 0.5, 0.0, 0.0]
        #     else:
        #         probs = [0.8, 0.2, 0.0, 0.0]
        #     j = tfp.distributions.Categorical(probs=probs).sample()
        #     float_data = float_data[:, :, j*240 : (j+1)*240]
        #     float_data.set_shape((None, None, 240))

        return BatchedData(
            inputs=float_data,
            inputs_length=tf.cast(batch[0][-2], tf.int32),
            inputs_filenames=batch[0][-1],
            labels=tf.cast(batch[1][0], tf.int32),
            labels_length=tf.cast(batch[1][1], tf.int32),
            labels_filenames=batch[1][2],
            payload=payload,
        )
    elif num_streams == 2:

        return BatchedData(
            inputs=(batch[0][0], batch[1][0]),
            inputs_length=(tf.cast(batch[0][-2], tf.int32), tf.cast(batch[1][-2], tf.int32)),
            inputs_filenames=(batch[0][-1], batch[1][-1]),
            labels=tf.cast(batch[2][0], tf.int32),
            labels_length=tf.cast(batch[2][1], tf.int32),
            labels_filenames=batch[2][2],
            payload=payload,
        )

    else:
        raise Exception('No implementation for num_streams > 2')


def advance_iterator(iterator):
    optional = tf.data.experimental.get_next_as_optional(iterator)

    def empty_or_zeros(shape, dtype):
        empty_shape = [0 if s is None else s for s in shape]
        return tf.zeros(empty_shape, dtype=dtype)

    if optional.has_value():
        batch = optional.get_value()
        is_done = False
    else:
        batch = tf.nest.map_structure(empty_or_zeros, iterator.output_shapes, iterator.output_types)
        is_done = True

    return batch, is_done


def get_files(file_list, dataset_dir, shuffle_sentences=False):
    with open(file_list, 'r') as f:
        contents = f.read().splitlines()

    contents = [path.join(dataset_dir, line.split()[0]) for line in contents]

    if shuffle_sentences is True:
        shuffle(contents)

    return contents
