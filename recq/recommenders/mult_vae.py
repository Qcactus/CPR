import tensorflow as tf
import numpy as np
import scipy.sparse
from tensorflow.contrib.layers import apply_regularization, l2_regularizer
from recq.utils.data import batch_iterator
from recq.utils.tf_utils import init_variables, save_model
from recq.utils.evaluator import create_evaluators
from recq.utils.early_stopping import EarlyStopping
from recq.tools.monitor import Timer
from recq.tools.io import print_seperate_line


class MultVAE(object):
    def __init__(self, args, dataset):
        tf.set_random_seed(args.seed)

        self.timer = Timer()
        self.dataset = dataset

        self.sess = tf.Session(
            config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
        )

        # self.p_dims = args.p_dims
        # if args.q_dims is None:
        #     self.q_dims = args.p_dims[::-1]
        # else:
        #     assert args.q_dims[0] == args.p_dims[
        #         -1], "Input and output dimension must equal each other for autoencoders."
        #     assert args.q_dims[-1] == args.p_dims[
        #         0], "Latent dimension for p- and q-network mismatches."
        #     self.q_dims = args.q_dims
        self.p_dims = args.weight_sizes + [dataset.n_item]
        self.q_dims = self.p_dims[::-1]
        self.dims = self.q_dims + self.p_dims[1:]

        self.construct_placeholders()

        self.logits_var, self.loss_var, self.train_op_var = self.build_graph(args)
        self.saver = tf.train.Saver(max_to_keep=1)

    def construct_placeholders(self):
        self.input_ph = tf.placeholder(dtype=tf.float32, shape=[None, self.dims[0]])
        self.keep_prob_ph = tf.placeholder_with_default(1.0, shape=None)

        # placeholders with default values when scoring
        self.is_training_ph = tf.placeholder_with_default(0.0, shape=None)
        self.anneal_ph = tf.placeholder_with_default(1.0, shape=None)

    def build_graph(self, args):
        self._construct_weights()

        logits, KL = self.forward_pass()
        log_softmax_var = tf.nn.log_softmax(logits)

        neg_ll = -tf.reduce_mean(
            tf.reduce_sum(log_softmax_var * self.input_ph, axis=-1)
        )
        # apply regularization to weights
        reg = l2_regularizer(args.reg)

        reg_var = apply_regularization(reg, self.weights_q + self.weights_p)
        # tensorflow l2 regularization multiply 0.5 to the l2 norm
        # multiply 2 so that it is back in the same scale
        neg_ELBO = neg_ll + self.anneal_ph * KL + 2 * reg_var

        train_op = tf.train.AdamOptimizer(args.lr).minimize(neg_ELBO)

        return logits, neg_ELBO, train_op

    def q_graph(self):
        mu_q, std_q, KL = None, None, None

        h = tf.nn.l2_normalize(self.input_ph, 1)
        h = tf.nn.dropout(h, self.keep_prob_ph)

        for i, (w, b) in enumerate(zip(self.weights_q, self.biases_q)):
            h = tf.matmul(h, w) + b

            if i != len(self.weights_q) - 1:
                h = tf.nn.tanh(h)
            else:
                mu_q = h[:, : self.q_dims[-1]]
                logvar_q = h[:, self.q_dims[-1] :]

                std_q = tf.exp(0.5 * logvar_q)
                KL = tf.reduce_mean(
                    tf.reduce_sum(
                        0.5 * (-logvar_q + tf.exp(logvar_q) + mu_q ** 2 - 1), axis=1
                    )
                )
        return mu_q, std_q, KL

    def p_graph(self, z):
        h = z

        for i, (w, b) in enumerate(zip(self.weights_p, self.biases_p)):
            h = tf.matmul(h, w) + b

            if i != len(self.weights_p) - 1:
                h = tf.nn.tanh(h)
        return h

    def forward_pass(self):
        # q-network
        mu_q, std_q, KL = self.q_graph()
        epsilon = tf.random_normal(tf.shape(std_q))

        sampled_z = mu_q + self.is_training_ph * epsilon * std_q

        # p-network
        logits = self.p_graph(sampled_z)

        return logits, KL

    def _construct_weights(self):
        self.weights_q, self.biases_q = [], []

        for i, (d_in, d_out) in enumerate(zip(self.q_dims[:-1], self.q_dims[1:])):
            if i == len(self.q_dims[:-1]) - 1:
                # we need two sets of parameters for mean and variance,
                # respectively
                d_out *= 2
            weight_key = "weight_q_{}to{}".format(i, i + 1)
            bias_key = "bias_q_{}".format(i + 1)

            self.weights_q.append(
                tf.get_variable(
                    name=weight_key,
                    shape=[d_in, d_out],
                    initializer=tf.contrib.layers.xavier_initializer(),
                )
            )

            self.biases_q.append(
                tf.get_variable(
                    name=bias_key,
                    shape=[d_out],
                    initializer=tf.truncated_normal_initializer(stddev=0.001),
                )
            )

        self.weights_p, self.biases_p = [], []

        for i, (d_in, d_out) in enumerate(zip(self.p_dims[:-1], self.p_dims[1:])):
            weight_key = "weight_p_{}to{}".format(i, i + 1)
            bias_key = "bias_p_{}".format(i + 1)
            self.weights_p.append(
                tf.get_variable(
                    name=weight_key,
                    shape=[d_in, d_out],
                    initializer=tf.contrib.layers.xavier_initializer(),
                )
            )

            self.biases_p.append(
                tf.get_variable(
                    name=bias_key,
                    shape=[d_out],
                    initializer=tf.truncated_normal_initializer(stddev=0.001),
                )
            )

    def fit(self, args, model_dir):

        init_variables(self.sess, self.saver, args.load_model, model_dir)

        train_data = scipy.sparse.csr_matrix(
            (
                np.ones_like(self.dataset.u_interacts),
                (self.dataset.u_interacts, self.dataset.i_interacts),
            ),
            dtype=np.float32,
            shape=(self.dataset.n_user, self.dataset.n_item),
        )

        # Create evaluators and early_stopping.
        self.evaluators = {}
        if args.eval_epoch is not None:
            self.evaluators = create_evaluators(
                self.dataset, args.eval_types, args.metrics, args.ks, args.n_thread
            )
            self.early_stopping = EarlyStopping(args.early_stop)

        # Start training and evaluation.
        print_seperate_line()
        if args.eval_epoch is not None:
            self.eval(train_data, args)
            print_seperate_line()

        self.update_count = 0.0
        for epoch in range(1, args.epoch + 1):
            self.train_1_epoch(train_data, epoch, args)

            if args.eval_epoch is not None and epoch % args.eval_epoch == 0:
                print_seperate_line()
                self.eval(train_data, args)
                print_seperate_line()

                if self.early_stopping.check_stop(self.evaluators, epoch):
                    break

        print(self.early_stopping)
        print_seperate_line()

        # Save model.
        if args.save_model:
            save_model(self.sess, self.saver, args.verbose_name, args.epoch, model_dir)

    def train_1_epoch(self, train_data, epoch, args):
        self.timer.start("Epoch {}".format(epoch))

        losses = []
        for _ in range(self.dataset.n_user // args.batch_size):
            users = np.random.choice(range(self.dataset.n_user), size=args.batch_size)
            X = train_data[users].toarray()

            if args.total_anneal_steps > 0:
                anneal = min(
                    args.anneal_cap, 1.0 * self.update_count / args.total_anneal_steps
                )
            else:
                anneal = args.anneal_cap

            feed_dict = {
                self.input_ph: X,
                self.keep_prob_ph: 0.5,
                self.anneal_ph: anneal,
                self.is_training_ph: 1,
            }
            batch_loss, _ = self.sess.run(
                [self.loss_var, self.train_op_var], feed_dict=feed_dict
            )
            losses.append(batch_loss)
            self.update_count += 1

        self.timer.stop(
            "loss = {:.5f}, anneal = {:.5f}".format(np.mean(losses), anneal)
        )

    def eval(self, train_data, args):
        self.timer.start("Evaluation")
        for evaluator in self.evaluators.values():
            for idx, batch_u in enumerate(
                batch_iterator(evaluator.eval_users, args.eval_batch_size)
            ):
                batch_users_idx = range(
                    idx * args.eval_batch_size,
                    idx * args.eval_batch_size + len(batch_u),
                )
                X = train_data[batch_u].toarray()
                batch_ratings = self.sess.run(
                    self.logits_var, feed_dict={self.input_ph: X}
                )

                for idx, user in enumerate(batch_u):
                    batch_ratings[idx][self.dataset.train[user]] = -np.inf

                evaluator.update(batch_ratings, batch_users_idx)

            evaluator.update_final()
            print(evaluator)

        self.timer.stop()
