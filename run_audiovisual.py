import os
from absl import app, flags
from avsr import utils
from avsr import run_experiment

FLAGS = flags.FLAGS

def main(argv):

    os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu_id
    FLAGS.architecture = 'av_transformer'
    records_path = './data/'

    video_train_record = records_path + 'rgb36lips_train_success_aus.tfrecord'
    video_test_record = records_path + 'rgb36lips_test_success_aus.tfrecord'
    labels_train_record = records_path + 'characters_train_success.tfrecord'
    labels_test_record = records_path + 'characters_test_success.tfrecord'

    audio_train_records = (
        records_path + 'logmel_train_success_clean.tfrecord',
        records_path + 'logmel_train_success_cafe_10db.tfrecord',
        records_path + 'logmel_train_success_cafe_0db.tfrecord',
        records_path + 'logmel_train_success_cafe_-5db.tfrecord'
    )
    audio_test_records = (
        records_path + 'logmel_test_success_clean.tfrecord',
        records_path + 'logmel_test_success_cafe_10db.tfrecord',
        records_path + 'logmel_test_success_cafe_0db.tfrecord',
        records_path + 'logmel_test_success_cafe_-5db.tfrecord'
    )

    iterations = (
        (100, 20, 'clean'),
        (100, 20, '10db'),
        (100, 20, '0db'),
        (100, 20, '-5db')
    )

    learning_rates = (
        (0.001, 0.0001),  # clean
        (0.001, 0.0001),  # 10db
        (0.001, 0.0001),  # 0db
        (0.001, 0.0001)  # -5db
    )

    run_experiment(
        video_train_record=video_train_record,
        video_test_record=video_test_record,
        labels_train_record=labels_train_record,
        labels_test_record=labels_test_record,
        audio_train_records=audio_train_records,
        audio_test_records=audio_test_records,
        iterations=iterations,
        learning_rates=learning_rates,
    )


if __name__ == '__main__':
    utils.avsr_flags()
    app.run(main)

