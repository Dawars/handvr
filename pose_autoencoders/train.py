import tensorflow as tf

from os.path import join
from pose_autoencoders.deep_ae import deep_ae_model_fn
from pose_autoencoders.pose_loader import get_poses

FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_enum('type', 'vanilla', ['vanilla', 'conv', 'vae', 'vae_conv'], 'Type of model.')
tf.flags.DEFINE_enum('optimizer', 'adam', ['adam', 'rmsprop'], 'Optimizer.')
tf.flags.DEFINE_enum('loss', 'mse', ['mse'], 'Loss function.')
tf.flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')
tf.flags.DEFINE_integer('max_steps', 2000, 'Number of steps to run trainer.')
tf.flags.DEFINE_list('encoder_units', [32, 16, 2], 'Number of units in encoder per layer.')
tf.flags.DEFINE_list('decoder_units', [16, 32], 'Number of units in decoder per layer.')
tf.flags.DEFINE_integer('batch_size', 32, 'Batch size.')
tf.flags.DEFINE_string('model_dir', './model', 'Directory to put the training data.')
tf.flags.DEFINE_string('log_dir', './logs', 'Directory to put the log data.')


def main(argv):
    save_dir = get_save_name()

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True  # do not allocate all the vram

    model_params = {}

    if FLAGS.type == 'vanilla':
        nn = tf.estimator.Estimator(model_fn=deep_ae_model_fn, params=model_params,
                                    model_dir=join(FLAGS.model_dir, save_dir),
                                    config=tf.estimator.RunConfig(session_config=config))

    summary_hook = tf.train.SummarySaverHook(
        save_steps=100,
        output_dir=join(FLAGS.log_dir, save_dir),
        summary_op=tf.summary.merge_all(),
        scaffold=tf.train.Scaffold()
    )
    # Train
    nn.train(input_fn=get_poses(), steps=FLAGS.max_steps, hooks=[summary_hook])

    ev = nn.evaluate(input_fn=get_poses())
    print("Loss: %s" % ev["loss"])
    print("Root Mean Squared Error: %s" % ev["rmse"])


def get_save_name():
    encoder = FLAGS.encoder_units
    decoder = FLAGS.decoder_units
    return "{}_{}_{}_{}_{}_{}_{}" \
        .format(FLAGS.type, FLAGS.learning_rate, FLAGS.batch_size, encoder, decoder, FLAGS.optimizer, FLAGS.loss)


if __name__ == '__main__':
    tf.app.run(main)
