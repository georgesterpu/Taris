import tensorflow as tf
from .io_utils import make_bimodal_dataset, make_unimodal_dataset, structure_data, create_unit_dict, advance_iterator
from .metrics import compute_wer, write_sequences_to_labelfile
from os import path, makedirs
import numpy as np
import time
from .loss import make_loss as loss_fn, regularisation_loss
from .optimiser import LearningRateSchedule
from .visualise.segmentation import get_segmentation_timestamps, write_praat_intensity, \
    gen_len_hists, plot_histograms, average_dict_values

from absl import flags

assert tf.__version__.startswith('2')
FLAGS = flags.FLAGS
flags.DEFINE_string('unit_list_file', './avsr/misc/character_list', 'File containing the list of graphemes')
flags.DEFINE_string('unit', 'character', 'Modelled linguistic unit')


class AVSR(object):
    def __init__(self,
                 video_train_record=None,
                 video_test_record=None,
                 audio_train_record=None,
                 audio_test_record=None,
                 labels_train_record=None,
                 labels_test_record=None,):

        self._unit_dict = create_unit_dict(unit_file=FLAGS.unit_list_file)

        self._video_train_record = video_train_record
        self._audio_train_record = audio_train_record
        self._labels_train_record = labels_train_record
        
        self._video_test_record = video_test_record
        self._audio_test_record = audio_test_record
        self._labels_test_record = labels_test_record

        self._create_model()
        self._create_datasets()
        self._create_optimiser()

    def _create_datasets(self):
        if self.num_streams == 1:
            self._train_dataset, self._dataset_info = make_unimodal_dataset(
                data_record=self._video_train_record if FLAGS.input_modality == 'video' else self._audio_train_record,
                label_record=self._labels_train_record,
                unit_list_file=FLAGS.unit_list_file,
                batch_size=FLAGS.batch_size,
                shuffle=True,
                bucket_width=45)
            self._test_dataset, self._dataset_info = make_unimodal_dataset(
                data_record=self._video_test_record if FLAGS.input_modality == 'video' else self._audio_test_record,
                label_record=self._labels_test_record,
                unit_list_file=FLAGS.unit_list_file,
                batch_size=FLAGS.batch_size,
                bucket_width=45,
                shuffle=False
            )

        if self.num_streams == 2:
            self._train_dataset, self._dataset_info = make_bimodal_dataset(
                video_record=self._video_train_record,
                audio_record=self._audio_train_record,
                label_record=self._labels_train_record,
                unit_list_file=FLAGS.unit_list_file,
                batch_size=FLAGS.batch_size,
                shuffle=True,
                bucket_width=45,
            )
            self._test_dataset, self._dataset_info = make_bimodal_dataset(
                video_record=self._video_test_record,
                audio_record=self._audio_test_record,
                label_record=self._labels_test_record,
                unit_list_file=FLAGS.unit_list_file,
                batch_size=FLAGS.batch_size,
                shuffle=False,
                bucket_width=45,
            )

    def _create_model(self):
        if FLAGS.architecture in ('transformer', ):
            from .transformer.model import Transformer
            self.model = Transformer(
                unit_dict=self._unit_dict,
            )
            self.num_streams = 1
        elif FLAGS.architecture in ('av_transformer',):
            from .transformer.model import AVTransformer
            self.model = AVTransformer(
                unit_dict=self._unit_dict
            )
            self.num_streams = 2
        else:
            raise ValueError('Architecture currently unsupported: {}'.format(FLAGS.architecture))

    @tf.function
    def _train_step(self, iterator):
        batch, is_done = advance_iterator(iterator)

        if is_done:
            return 0., 0., True
        data = structure_data(
                    batch,
                    num_streams=self.num_streams,
                    info=self._dataset_info)

        with tf.GradientTape() as tape:
            pred_output, _ = self.model(data, training=True)

            main_loss = loss_fn(logits=pred_output, label_ids=data.labels, sequence_lengths=data.labels_length)
            reg_loss = regularisation_loss(self.model.weights)
            extra_loss = tf.add_n(self.model.losses + [0.])
            loss = main_loss + extra_loss + reg_loss

            if FLAGS.transformer_dtype == 'float16':
                loss = self.optimiser.get_scaled_loss(loss)

        tvars = list({id(v): v for v in self.model.trainable_variables}.values())
        grads = tape.gradient(loss, tvars)
        if FLAGS.transformer_dtype == 'float16':
            grads = self.optimiser.get_unscaled_gradients(grads)
        clipped_grads, norm = tf.clip_by_global_norm(grads, FLAGS.max_gradient_norm)
        grads_and_vars = zip(clipped_grads, self.model.trainable_variables)
        self.optimiser.apply_gradients(grads_and_vars)

        return loss, extra_loss, False

    def train(
            self,
            target_epoch,
            learning_rate,
            initial_checkpoint=None,
            iteration_name='default',
            try_restore_from_prev_run=False):

        # tf.config.optimizer.set_jit(True)
        if FLAGS.use_tensorboard:
            writer = _initialise_summary(FLAGS.logfile, file_suffix=iteration_name)

        lr_schedule = LearningRateSchedule(
            initial_learning_rate=learning_rate,
            warmup_steps=FLAGS.lr_warmup_steps,
            initial_global_step=self.optimiser.iterations.numpy())
        self.optimiser.learning_rate = lr_schedule

        checkpoint = _init_checkpoint(self.model)
        checkpoint_manager = _init_checkpoint_manager(
            checkpoint=checkpoint,
            iteration_name=iteration_name
        )
        save_path = None
        restored = False
        last_epoch = 0
        if try_restore_from_prev_run:
            try:
                checkpoint_dir = path.join('checkpoints', FLAGS.logfile, iteration_name)
                prev_ckp = tf.train.latest_checkpoint(checkpoint_dir)
                last_epoch = int(prev_ckp.split('-')[-1])
                checkpoint.restore(prev_ckp)
                print('Restoring checkpoint from epoch {} from a previous run\n'.format(last_epoch))
                save_path = prev_ckp
                restored = True
            except Exception:
                print('Could not restore from checkpoint, training from scratch!\n')
        if not restored and initial_checkpoint is not None:
            try:
                last_iteration_name = initial_checkpoint.split('/')[-2]
                if last_iteration_name == iteration_name:
                    last_epoch = int(initial_checkpoint.split('-')[-1])
                checkpoint.restore(initial_checkpoint)
                print('Restoring checkpoint {}\n'.format(initial_checkpoint))
                save_path = initial_checkpoint
            except Exception:
                print('Could not restore from checkpoint, training from scratch!\n')

        if last_epoch >= target_epoch:
            print('Skipping iterations on {} speech at {} learning rate'.format(iteration_name, learning_rate))
            return save_path

        logfile = path.join('./logs/', FLAGS.logfile)
        makedirs(path.dirname(logfile), exist_ok=True)
        f = open(logfile, 'a')
        f.write(30*'=' + '\n')
        # self.dump_hparams_to_logfile(f)  # TODO pretty formatting
        f.write('Training for {} iterations at a {} learning rate on {} speech \n'
                .format(target_epoch - last_epoch, learning_rate, iteration_name))
        f.write(15 * '=' + '\n')

        if FLAGS.profiling is True:
            # tf.python.eager.profiler.start_profiler_server(6009)
            tf.summary.trace_on(profiler=True)

        for epoch in range(last_epoch + 1, target_epoch + 1):

            sum_loss = 0.0
            sum_wloss = 0.0
            start = time.time()

            batch_id = 0
            iterator = self._train_dataset.__iter__()

            while True:
                if FLAGS.use_tensorboard:
                    with writer.as_default():
                        loss, wloss, is_done = self._train_step(iterator)
                else:
                    loss, wloss, is_done = self._train_step(iterator)

                if is_done:
                    break

                sum_loss += loss.numpy()
                sum_wloss += wloss.numpy()

                print('Batch {}, loss {}'.format(batch_id, loss.numpy()))

                batch_id += 1

            print('epoch time: {}'.format(time.time() - start))
            avg_loss = sum_loss / batch_id
            avg_wc_loss = sum_wloss / batch_id / FLAGS.wordloss_weight
            print('Average epoch loss: {}'.format(avg_loss))
            f.write('Average batch loss at epoch {} is {} (word count loss: {})\n'.format(epoch, avg_loss, avg_wc_loss))
            f.flush()
            if FLAGS.use_tensorboard:
                with writer.as_default():
                    tf.summary.scalar('Average Batch loss', avg_loss, step=self.optimiser.iterations.numpy())
                    tf.summary.scalar('Average Word Count loss', avg_wc_loss, step=self.optimiser.iterations.numpy())

            if epoch % 10 == 0:
                save_path = checkpoint_manager.save(checkpoint_number=epoch)
                error_rate = self.evaluate(checkpoint_path=save_path, epoch=epoch, iteration_name=iteration_name)
                for (k, v) in error_rate.items():
                    f.write(k + ': {:.4f}% '.format(v))
                    if FLAGS.use_tensorboard:
                        with writer.as_default():
                            tf.summary.scalar(k + ' error rate', v, step=self.optimiser.iterations.numpy())
                f.write('\n')
                f.flush()

        if FLAGS.profiling is True:
            makedirs('/tmp/tf_profile/', exist_ok=True)
            tf.summary.trace_export(name='run1', profiler_outdir='/tmp/tf_profile/')

        return save_path

    @tf.function
    def _evaluate_step(self, next_fun):

        # batch, is_done = advance_iterator(iterator)
        batch = next_fun()

        data = structure_data(
            batch,
            num_streams=self.num_streams,
            info=self._dataset_info)

        pred_output, extra = self.model(data, training=False)
        extra['losses'] = self.model.losses
        return pred_output, data, extra

    def evaluate(
            self,
            checkpoint_path,
            epoch=None,
            outdir='./results/',
            iteration_name='default',
    ):

        checkpoint = _init_checkpoint(self.model)
        checkpoint.restore(checkpoint_path)

        # if FLAGS.use_tensorboard:
        #     writer = _initialise_summary(FLAGS.logfile, file_suffix=iteration_name + '_eval')

        predictions_dict = {}
        labels_dict = {}
        tmp_dict = {}
        bins = np.arange(0, 990, 30)
        ref_hist = np.zeros(len(bins)-1)
        hyp_hist = np.zeros(len(bins)-1)

        iterator = self._test_dataset.__iter__()

        def next_fun():
            return iterator.__next__()

        batch_id = 0
        sum_wloss = 0.0
        while True:
            print(batch_id)

            try:
                # if FLAGS.use_tensorboard:
                #     with writer.as_default():
                #         output, data, extra = self._evaluate_step(next_fun)
                # else:
                output, data, extra = self._evaluate_step(next_fun)
            except tf.errors.OutOfRangeError:
                break

            if FLAGS.architecture in ('transformer', 'av_transformer'):
                predicted_ids_batch = output['outputs'].numpy()
            elif FLAGS.architecture in ('online_segmental', 'experimental_segmental'):
                predicted_ids_batch = tf.argmax(output, axis=-1).numpy()
            elif FLAGS.architecture == 'online':
                predicted_ids_batch = output.numpy()
            else:
                predicted_ids_batch = output.predicted_ids[:, :, 0].numpy()  # select top (first) beam

            for idx in range(predicted_ids_batch.shape[0]):
                file = data.labels_filenames[idx].numpy().decode('utf-8')

                predicted_ids = predicted_ids_batch[idx, :]
                predicted_symbs = [self._unit_dict[sym] for sym in predicted_ids]
                predictions_dict[file] = predicted_symbs

                label_ids = data.labels[idx, :].numpy()
                label_symbs = [self._unit_dict[sym] for sym in label_ids]
                labels_dict[file] = label_symbs

                if FLAGS.write_halting_history:
                    fname = path.join(outdir, 'segmentations', FLAGS.logfile, iteration_name,
                                      file + '_tmp.praat')
                    makedirs(path.dirname(fname), exist_ok=True)

                    logits = extra['word_logits'].numpy()[idx, :, 0]
                    x, y = get_segmentation_timestamps(logits)
                    write_praat_intensity(x, y, fname.replace('_tmp', '_logits'))

                    cumsum = extra['word_floor_cumsum'].numpy()[idx, :, 0]
                    x, y = get_segmentation_timestamps(cumsum)
                    write_praat_intensity(x, y, fname.replace('_tmp', '_cumsum'))

                    ## TODO
                    tmp_dict[file], hyp_lens, ref_lens = gen_len_hists(file, x, y)
                    hyp_hist += np.histogram(hyp_lens, bins)[0]
                    ref_hist += np.histogram(ref_lens, bins)[0]

            if isinstance(extra['losses'], list):
                sum_wloss += extra['losses'][-1].numpy()
            else:
                sum_wloss += extra['losses'].numpy()

            batch_id += 1

        uer, uer_dict = compute_wer(predictions_dict, labels_dict)
        error_rate = {FLAGS.unit: uer * 100}
        if FLAGS.unit == 'character':
            wer, wer_dict = compute_wer(predictions_dict, labels_dict, split_words=True)
            error_rate['word'] = wer * 100
        if bool(tmp_dict):
            plot_histograms(
                hyp_hist, ref_hist, bins,
                fname=path.join(outdir, 'segmentations', FLAGS.logfile, iteration_name, 'word_hists_{}.png'.format(epoch)))
        error_rate['word_loss'] = sum_wloss / batch_id / FLAGS.wordloss_weight

        log_outdir = path.join('./predictions', FLAGS.logfile, iteration_name)
        makedirs(log_outdir, exist_ok=True)
        write_sequences_to_labelfile(
            predictions_dict,
            path.join(log_outdir, 'predicted_epoch_{}.mlf'.format(epoch)),
            labels_dict,
            uer_dict
        )

        return error_rate

    def _create_optimiser(self):
        if FLAGS.optimiser == 'lamb':
            import tensorflow_addons as tfa
            self.optimiser = tfa.optimizers.LAMB(
                learning_rate=0.0,
            )
        elif FLAGS.optimiser == 'radam':
            import tensorflow_addons as tfa
            self.optimiser = tfa.optimizers.RectifiedAdam(
                learning_rate=0.0,
            )
        elif FLAGS.optimiser == 'lookahead_radam':
            import tensorflow_addons as tfa
            optimiser = tfa.optimizers.RectifiedAdam(
                learning_rate=0.0)
            self.optimiser = tfa.optimizers.Lookahead(optimiser)
        else:
            self.optimiser = tf.keras.optimizers.Adam(
                learning_rate=0.0,  # safety measure
                # clipnorm=FLAGS.max_gradient_norm,  # wait for upstream fix
                amsgrad=FLAGS.amsgrad,
                beta_1=0.9,
                beta_2=0.999,
                epsilon=1e-7)

        if FLAGS.transformer_dtype == 'float16':
            self.policy = tf.keras.mixed_precision.experimental.Policy('mixed_float16')
            tf.keras.mixed_precision.experimental.set_policy(self.policy)
            self.optimiser = tf.keras.mixed_precision.experimental.LossScaleOptimizer(
                self.optimiser,
                self.policy.loss_scale)


def _init_checkpoint(model):
    return tf.train.Checkpoint(model=model)


def _initialise_summary(name, file_suffix=None, dirname='summaries'):
    summary_path = path.join(dirname, name)
    writer = tf.summary.create_file_writer(summary_path, flush_millis=10*1000, filename_suffix=file_suffix)
    return writer


def _init_checkpoint_manager(checkpoint, iteration_name):
    checkpoint_dir = path.join('checkpoints', FLAGS.logfile, iteration_name)

    manager = tf.train.CheckpointManager(
        checkpoint=checkpoint,
        directory=checkpoint_dir,
        max_to_keep=1)

    return manager
