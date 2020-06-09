from avsr import AVSR
from absl import flags

FLAGS = flags.FLAGS


def run_experiment(
        video_train_record=None,
        video_test_record=None,
        labels_train_record=None,
        labels_test_record=None,
        audio_train_records=None,
        audio_test_records=None,
        iterations=None,
        learning_rates=None,
        **kwargs):

    latest_ckp = None
    for lr, iters, audio_train, audio_test in zip(learning_rates, iterations, audio_train_records, audio_test_records):
        experiment = AVSR(
            audio_train_record=audio_train,
            audio_test_record=audio_test,
            video_train_record=video_train_record,
            video_test_record=video_test_record,
            labels_train_record=labels_train_record,
            labels_test_record=labels_test_record,
        )

        latest_ckp = experiment.train(
            target_epoch=iters[0],
            learning_rate=lr[0],  # high learning rate
            initial_checkpoint=latest_ckp,
            iteration_name=iters[2],
            try_restore_from_prev_run=True,
        )

        del experiment
        experiment = AVSR(
            audio_train_record=audio_train,
            audio_test_record=audio_test,
            video_train_record=video_train_record,
            video_test_record=video_test_record,
            labels_train_record=labels_train_record,
            labels_test_record=labels_test_record,
        )
        latest_ckp = experiment.train(
            target_epoch=iters[1] + iters[0],
            learning_rate=lr[1],  # low learning rate
            initial_checkpoint=latest_ckp,
            iteration_name=iters[2],
            try_restore_from_prev_run=True,
        )
        del experiment


def run_experiment_allsnr(
        video_train_record=None,
        video_test_record=None,
        labels_train_record=None,
        labels_test_record=None,
        audio_train_record=None,
        audio_test_record=None,
        iterations=None,
        learning_rates=None,
        **kwargs):

    latest_ckp = None
    for level, (lr, iters) in enumerate(zip(learning_rates, iterations)):

        FLAGS.noise_level = level

        experiment = AVSR(
            audio_train_record=audio_train_record,
            audio_test_record=audio_test_record,
            video_train_record=video_train_record,
            video_test_record=video_test_record,
            labels_train_record=labels_train_record,
            labels_test_record=labels_test_record,
        )

        latest_ckp = experiment.train(
            target_epoch=iters[0],
            learning_rate=lr[0],  # high learning rate
            initial_checkpoint=latest_ckp,
            iteration_name=iters[2],
            try_restore_from_prev_run=True,
        )
        del experiment

        experiment = AVSR(
            audio_train_record=audio_train_record,
            audio_test_record=audio_test_record,
            video_train_record=video_train_record,
            video_test_record=video_test_record,
            labels_train_record=labels_train_record,
            labels_test_record=labels_test_record,
        )
        latest_ckp = experiment.train(
            target_epoch=iters[1] + iters[0],
            learning_rate=lr[1],  # low learning rate
            initial_checkpoint=latest_ckp,
            iteration_name=iters[2],
            try_restore_from_prev_run=True,
        )
        del experiment
