import tensorflow as tf
import tensorflow_addons as tfa
from absl import flags

FLAGS = flags.FLAGS


class LearningRateSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    """Learning rate schedule.
    Forked from https://github.com/tensorflow/models/blob/master/official/transformer/v2/optimizer.py
    """

    def __init__(self, initial_learning_rate, warmup_steps, initial_global_step=0):
        """Initialize configuration of the learning rate schedule.

        Args:
          initial_learning_rate: A float, the initial learning rate.
          warmup_steps: An integer, the number of steps required for linear warmup.
          initial_global_step: An integer, the offset applied to the global step
            acting as a workaround for resetting the global step of the optimiser
        """
        super(LearningRateSchedule, self).__init__()
        self.initial_learning_rate = initial_learning_rate
        self.warmup_steps = tf.cast(warmup_steps, tf.float32)
        self.initial_global_step = initial_global_step

    def __call__(self, global_step):
        """Calculate learning rate with linear warmup and rsqrt decay.

        Args:
          global_step: An integer, the current global step used for learning rate
            calculation.

        Returns:
          A float, the learning rate needs to be used for current global step.
        """
        with tf.name_scope('learning_rate_schedule'):
            global_step = tf.cast(global_step, tf.float32)
            learning_rate = self.initial_learning_rate
            # Apply linear warmup
            learning_rate *= tf.minimum(1.0, (global_step + 1 - self.initial_global_step) / self.warmup_steps)

            return learning_rate

    def get_config(self):
        """Get the configuration of the learning rate schedule."""
        return {
            'initial_learning_rate': self.initial_learning_rate,
            'warmup_steps': self.warmup_steps,
        }


def init_optimiser(name):
    if name == 'lamb':
        optimiser = tfa.optimizers.LAMB(
            learning_rate=0.0,
        )
    elif name == 'novograd':
        optimiser = tfa.optimizers.NovoGrad(
            learning_rate=0.0,
        )
    elif name == 'radam':
        optimiser = tfa.optimizers.RectifiedAdam(
            learning_rate=0.0,
        )
    elif name == 'lookahead_radam':
        optimiser = tfa.optimizers.RectifiedAdam(
            learning_rate=0.0)
        optimiser = tfa.optimizers.Lookahead(optimiser)
    elif name == 'adam':
        optimiser = tf.keras.optimizers.Adam(
            learning_rate=0.0,  # safety measure
            # clipnorm=FLAGS.max_gradient_norm,  # wait for upstream fix
            amsgrad=FLAGS.amsgrad,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-7)
    else:
        raise ValueError('Unknown optimiser: {}'.format(name))

    return optimiser
