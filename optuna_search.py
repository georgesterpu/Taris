import os
from absl import app, flags
from avsr import utils
import optuna
from avsr import AVSR

FLAGS = flags.FLAGS


def objective(trial):
    wloss_weight = trial.suggest_uniform("wloss_weight", 0.0, 1.0)
    return wloss_weight

def main(argv):

    os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu_id
    FLAGS.architecture = 'transformer'
    records_path = './data/'

    labels_train_record = records_path + 'characters_train_success.tfrecord'
    labels_test_record = records_path + 'characters_test_success.tfrecord'

    audio_train_records = (
        records_path + 'logmel_train_success_clean.tfrecord',
    )
    audio_test_records = (
        records_path + 'logmel_test_success_clean.tfrecord',
    )

    def experiment(trial):
        FLAGS.wordloss_weight = trial.suggest_uniform("Wloss_weight", 0.0, 1.0)

        inst = AVSR(
            audio_train_record=audio_train_records[0],
            audio_test_record=audio_test_records[0],
            labels_train_record=labels_train_record,
            labels_test_record=labels_test_record,
        )
        acc = inst.optuna_train(target_epoch=20, learning_rate=0.001)
        return acc

    study = optuna.create_study()
    study.optimize(experiment, n_trials=100, show_progress_bar=True)


if __name__ == '__main__':
    utils.avsr_flags()
    app.run(main)

