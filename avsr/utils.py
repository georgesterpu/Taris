from absl import flags

FLAGS = flags.FLAGS


def avsr_flags():

    # Generic flags
    flags.DEFINE_integer('buffer_size', 15000, 'Shuffle buffer size')
    flags.DEFINE_integer('batch_size', 64, 'Batch Size')
    flags.DEFINE_integer('embedding_size', 128, 'Embedding dimension')
    flags.DEFINE_integer('beam_width', 10, 'Beam Width')
    flags.DEFINE_boolean('enable_function', False, 'Enable Function?')
    flags.DEFINE_string('architecture', 'transformer', 'Network Architecture')
    flags.DEFINE_string('gpu_id', '0', 'GPU index')
    flags.DEFINE_string('input_modality', 'audio', 'Switch between A and V inputs')

    flags.DEFINE_integer('noise_level', 0, 'Noise level in range {0, 1, 2, 3}')
    flags.DEFINE_boolean('mix_noise', False, 'TBA')

    flags.DEFINE_multi_integer('cnn_filters', (16, 32, 48, 64), 'Number of CNN filters per layer')
    flags.DEFINE_integer('cnn_dense_units', 256, 'Number of neurons in the CNN output layer')
    flags.DEFINE_string('cnn_activation', 'relu', 'Activation function in CNN layers')
    flags.DEFINE_string('cnn_final_activation', None, 'Activation function in the final CNN layer')
    flags.DEFINE_string('cnn_normalisation', 'layer_norm', 'Normalisation function in CNN blocks')
    flags.DEFINE_boolean('cnn_final_clip', True, 'Clip the activation of the final CNN layer')

    # flags.DEFINE_float('learning_rate', 0.001, 'Learning rate')
    flags.DEFINE_integer('lr_warmup_steps', 750, 'Number of steps for the Learning rate linear warmup')
    flags.DEFINE_float('max_gradient_norm', 1.0, 'Clip the global gradient norm')
    flags.DEFINE_string('optimiser', 'adam', 'Optimiser type')
    flags.DEFINE_boolean('amsgrad', False, 'Use AMSGrad ?')

    flags.DEFINE_string('logfile', 'default_logfile', 'Logfile name')
    flags.DEFINE_boolean('profiling', False, 'Enable profiling for TensorBoard')
    flags.DEFINE_boolean('write_halting_history', False, 'Dump segmentations to praat files')
    flags.DEFINE_boolean('plot_alignments', False, 'Write alignment summaries in TensorBoard')
    flags.DEFINE_boolean('use_tensorboard', False, 'Export TensorBoard summary')


    # RNN seq2seq FLAGS
    flags.DEFINE_string('encoder', 'RNN', 'Encoder type')
    flags.DEFINE_string('recurrent_activation', 'sigmoid', 'Activation function inside LSTM Cell')
    flags.DEFINE_string('encoder_input_normalisation', 'layer_norm', 'Normalisation function for the Encoder input')
    flags.DEFINE_string('cell_type', 'ln_lstm', 'Recurrent cell type')
    flags.DEFINE_multi_integer('encoder_units_per_layer', 3 * (256, ), 'Number of encoder cells in each recurrent layer')
    flags.DEFINE_multi_integer('decoder_units_per_layer', 1 * (256,), 'Number of decoder cells in each recurrent layer')
    flags.DEFINE_multi_float('dropout_probability', 3 * (0.1,), 'Dropout rate for for RNN cells')
    flags.DEFINE_multi_float('rnn_l1_l2_weight_regularisation', (0.0, 0.0001), 'Weight regularisation (L1 and L2) for RNN cells')
    flags.DEFINE_boolean('recurrent_regularisation', False, 'Use regularisation in the recurrent LSTM kernel')
    flags.DEFINE_boolean('regularise_all', True, 'Regularise all model variables ?')
    flags.DEFINE_string('recurrent_initialiser', None, 'Recurrent kernel initialiser')
    flags.DEFINE_boolean('recurrent_dropout', True, 'Apply dropout on recurrent state')
    flags.DEFINE_boolean('enable_attention', True, 'Enable Attention ?')
    flags.DEFINE_float('output_sampling', 0.1, 'Output Sampling Rate')
    flags.DEFINE_float('lstm_state_dropout', 0.1, 'Dropout applied to the h state of the LSTM')
    flags.DEFINE_string('decoder_initialisation', 'final_encoder_state', 'Decoder initialisation scheme')
    flags.DEFINE_string('segmental_variant', 'v1', 'Segmental model variant')

    # Transformer Model
    flags.DEFINE_integer('transformer_hidden_size', 256, 'State size of the Transformer layers')
    flags.DEFINE_integer('transformer_num_encoder_layers', 6, 'Number of layers in the encoder stack')
    flags.DEFINE_integer('transformer_num_decoder_layers', 6, 'Number of layers in the decoder stack')
    flags.DEFINE_integer('transformer_num_heads', 1, 'Number of attention_heads')
    flags.DEFINE_integer('transformer_filter_size', 256, 'Filter size')
    flags.DEFINE_float('transformer_relu_dropout', 0.1, 'Filter size')
    flags.DEFINE_float('transformer_attention_dropout', 0.1, 'Filter size')
    flags.DEFINE_float('transformer_layer_postprocess_dropout', 0.1, 'Post-processing layer dropout')
    flags.DEFINE_string('transformer_dtype', 'float32', 'Data type')
    flags.DEFINE_integer('transformer_extra_decode_length', 0, 'Extra Decode Length')
    flags.DEFINE_integer('transformer_beam_size', 10, 'Beam search width')
    flags.DEFINE_float('transformer_alpha', 0.6, 'Used for length normalisation in beam search')
    flags.DEFINE_boolean('transformer_online_encoder', False, 'Whether or not to use a causal attention bias')
    flags.DEFINE_integer('transformer_encoder_lookahead', 11, 'Number of frames for encoder attention lookahead')
    flags.DEFINE_integer('transformer_encoder_lookback', 11, 'Number of frames for encoder attention lookback')
    flags.DEFINE_boolean('transformer_online_decoder', False, 'Wheter or not to use online decoding')
    flags.DEFINE_integer('transformer_decoder_lookahead', 5, 'Number of segments for cross-modal attention lookahead')
    flags.DEFINE_integer('transformer_decoder_lookback', 5, 'Number of segments for cross-modal attention lookback')
    flags.DEFINE_float('transformer_l1_regularisation', 0.0, 'Transformer L1 weight regularisation')
    flags.DEFINE_float('transformer_l2_regularisation', 0.0001, 'Transformer L2 weight regularisation')

    ## Experimental flags
    flags.DEFINE_integer('transformer_num_avalign_layers', 1, 'Number of layers in the AVAlign stack')
    flags.DEFINE_boolean('au_loss', False, 'Use the Action Unit Loss for multimodal encoders')
    flags.DEFINE_float('au_loss_weight', 10.0, 'Scalar multiplier of the AU loss')
    flags.DEFINE_boolean('constrain_av_attention', False, 'Limit the search space of the A-V alignment'
                                                          ' to a window centred on the main diagonal')
    flags.DEFINE_integer('constrain_frames', 2, 'Num frames')
    flags.DEFINE_boolean('word_loss', False, 'Word counting loss')
    flags.DEFINE_float('wordloss_weight', 0.01, 'Num words loss')

    flags.DEFINE_string('wb_activation', 'sigmoid', 'Activation for the encoder halting unit')
