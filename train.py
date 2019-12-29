import tensorflow.compat.v1 as tf
import numpy as np
import data_helpers
from tensorflow.keras.preprocessing.text import Tokenizer
import tensorflow.keras.preprocessing.sequence as S
from text_cnn import TextCNN

# parameters
# =====================================

# data loading params
tf.flags.DEFINE_float('dev_sample_percentage', 0.1, '"Percentage of the training data to use for validation"')
tf.flags.DEFINE_string("positive_data_file", "./data/rt-polarity.pos", "Data source for the positive data")
tf.flags.DEFINE_string("negative_data_file", "./data/rt-polarity.neg", "Data source for the negative data")

# model hyper-parameters
tf.flags.DEFINE_integer("embedding_dim", 128, "Dimensionality of character embedding (default: 128)")
tf.flags.DEFINE_string("filter_sizes", "3,4,5", "Comma-separated filter sizes (default: '3,4,5')")
tf.flags.DEFINE_integer("num_filters", 64, "Number of filters per filter size (default: 64)")
tf.flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability (default: 0.5)")
tf.flags.DEFINE_float("l2_reg_lambda", 0.0, "L2 regularization lambda (default: 0.0)")

# training parameters
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("num_epochs", 2000, "Number of training epochs (default: 2000)")
tf.flags.DEFINE_integer("evaluate_every", 100, "Evaluate model on dev set after this many steps (default: 100)")
tf.flags.DEFINE_integer("checkpoint_every", 5, "Save model after this many steps (default: 5)")

# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

# parse analysis
FLAGS = tf.flags.FLAGS

# print all parameters
# print("\nParameters:")
# for attr, value in sorted(FLAGS.flag_values_dict().items()):
#     print("{}={}".format(attr.upper(), value))
# print("")


def pre_process():
    # data preparation
    # ====================================

    # load data
    print('Loading data...')
    x_text, y = data_helpers.load_data_and_labels(FLAGS.positive_data_file, FLAGS.negative_data_file)

    # build vocabulary
    max_document_length = max([len(x.split(' ')) for x in x_text])
    tokenizer = Tokenizer(num_words=None)
    tokenizer.fit_on_texts(x_text)
    sequences = tokenizer.texts_to_sequences(x_text)
    x = S.pad_sequences(sequences, maxlen=max_document_length, padding='post')

    # random shuffle data
    np.random.seed(10)
    shuffle_indices = np.random.permutation(np.arange(len(y)))
    x_shuffled = x[shuffle_indices]
    y_shuffled = y[shuffle_indices]

    # Split train/test set
    dev_sample_index = -1 * int(FLAGS.dev_sample_percentage * float(len(y)))
    x_train, x_dev = x_shuffled[:dev_sample_index], x_shuffled[dev_sample_index:]
    y_train, y_dev = y_shuffled[:dev_sample_index], y_shuffled[dev_sample_index:]

    del x, y, x_shuffled, y_shuffled

    print("Vocabulary Size: {}".format(len(tokenizer.word_index)))
    print("Train/Dev split: {:d}/{:d}".format(len(y_train), len(y_dev)))
    return x_train, y_train, tokenizer, x_dev, y_dev


def train(x_train, y_train, tokenizer, x_dev, y_dev):
    with tf.Graph().as_default():
        session_conf = tf.ConfigProto(
            allow_soft_placement=FLAGS.allow_soft_placement,
            log_device_placement=FLAGS.log_device_placement)
        sess = tf.Session(config=session_conf)
        with sess.as_default():
            cnn = TextCNN(
                sequence_length=x_train.shape[1],
                num_classes=y_train.shape[1],
                vocab_size=len(tokenizer.word_index),
                embedding_size=FLAGS.embedding_dim,
                filter_sizes=list(map(int, FLAGS.filter_sizes.split(","))),
                num_filters=FLAGS.num_filters,
                l2_reg_lambda=FLAGS.l2_reg_lambda)
        optimizer = tf.train.AdamOptimizer(1e-3).minimize(cnn.loss)
        sess.run(tf.global_variables_initializer())

        # training loop
        for i in range(FLAGS.num_epochs):
            x_batch, y_batch = data_helpers.next_batch(FLAGS.batch_size, x_train, y_train)
            feed_dict_train = {cnn.input_x: x_batch, cnn.input_y: y_batch, cnn.dropout_keep_prob: FLAGS.dropout_keep_prob}
            _, loss, accuracy = sess.run([optimizer, cnn.loss, cnn.accuracy], feed_dict=feed_dict_train)
            if i % 10 == 0:
                print('step {}:, loss: {}, accuracy: {}'.format(i, loss, accuracy))

        # valid step
        feed_dict_valid = {cnn.input_x: x_dev, cnn.input_y: y_dev, cnn.dropout_keep_prob: 1.}
        print('valid accuracy: {}'.format(sess.run(cnn.accuracy, feed_dict=feed_dict_valid)))


if __name__ == '__main__':
    x_train, y_train, tokenizer, x_dev, y_dev = pre_process()
    train(x_train, y_train, tokenizer, x_dev, y_dev)




