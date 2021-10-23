import os
import tensorflow as tf
import numpy as np
from recq.tools.io import mkdir


def sp_mat_2_sp_tensor(X):
    """Convert a scipy sparse matrix to tf.SparseTensor.

    Returns:
        tf.SparseTensor: SparseTensor after conversion.

    """
    coo = X.tocoo().astype(np.float32)
    indices = np.mat([coo.row, coo.col]).transpose()
    return tf.SparseTensor(indices, coo.data, coo.shape)


def init_variables(sess, saver, load_model, model_dir):
    # In case that the saved model doesn't include all variables, init all variables first.
    sess.run(tf.global_variables_initializer())
    if load_model is not None:
        try:
            saver.restore(sess, os.path.join(model_dir, load_model))
            print("Load model: {}".format(load_model))
        except:
            raise IOError(
                "Failed to find any matching files for model {0}.".format(load_model)
            )


def save_model(sess, saver, verbose_name, epoch, model_dir):
    mkdir(model_dir)
    filename = verbose_name + "_epoch_{}".format(epoch)
    path = os.path.join(model_dir, filename)
    saver.save(sess=sess, save_path=path)
    print("Model saved to {}".format(path))
