import tensorflow as tf

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
from time import time
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import numpy as np
from recq.utils.data import batch_iterator, DICESampler
from recq.utils.graph import create_norm_adj, create_ngcf_embed, create_lightgcn_embed
from recq.utils.inference import inner_product, mlp
from recq.utils.loss import bpr_loss, mask_bpr_loss, l2_embed_loss, discrepency_loss
from recq.utils.tf_utils import init_variables, save_model
from recq.utils.evaluator import create_evaluators
from recq.utils.early_stopping import EarlyStopping
from recq.tools.monitor import Timer
from recq.tools.io import print_seperate_line


class DICE(object):
    """LightGCN model

    SIGIR 2020. He, Xiangnan, Kuan Deng, Xiang Wang, Yan Li, Yongdong Zhang, and Meng Wang.
    "LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation." arXiv
    preprint arXiv:2002.02126 (2020).
    """

    def __init__(self, args, dataset):
        """Initializing the model. Create parameters, placeholders, embeddings and loss function."""
        tf.set_random_seed(args.seed)

        self.timer = Timer()
        self.dataset = dataset

        self.sess = tf.Session(
            config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
        )

        self._build_graph(args)
        self.saver = tf.train.Saver(max_to_keep=1)

    def _build_graph(self, args):
        self.initializer = tf.contrib.layers.xavier_initializer()
        self.create_variables(args)
        self.create_embeds(args)
        self.create_batch_ratings(args)
        self.sampler = DICESampler(self.dataset, args)
        self.int_weight = tf.placeholder(tf.float32, shape=())
        self.pop_weight = tf.placeholder(tf.float32, shape=())
        self.dis_pen = args.dis_pen
        self.create_loss(args)
        self.opt = tf.train.AdamOptimizer(learning_rate=args.lr).minimize(self.loss)

    def fit(self, args, model_dir):

        init_variables(self.sess, self.saver, args.load_model, model_dir)

        # Create evaluators and early_stoppings
        # evaluators: for different eval_set
        self.evaluators = {}

        if args.eval_epoch is not None:
            self.evaluators = create_evaluators(
                self.dataset, args.eval_types, args.metrics, args.ks, args.n_thread
            )
            self.early_stopping = EarlyStopping(args.early_stop)

        # Start training and evaluation.
        print_seperate_line()
        if args.eval_epoch is not None:
            self.eval(args)
            print_seperate_line()

        self.int_weight_v = args.int_weight
        self.pop_weight_v = args.pop_weight

        for epoch in range(1, args.epoch + 1):
            self.train_1_epoch(epoch, args)
            self.int_weight_v *= args.loss_decay
            self.pop_weight_v *= args.loss_decay
            self.sampler.margin *= args.margin_decay

            if args.eval_epoch is not None and epoch % args.eval_epoch == 0:
                print_seperate_line()
                self.eval(args)
                print_seperate_line()

                if self.early_stopping.check_stop(self.evaluators, epoch):
                    break

        print(self.early_stopping)
        print_seperate_line()

        # Save model.
        if args.save_model:
            save_model(self.sess, self.saver, args.verbose_name, args.epoch, model_dir)

    def create_variables(self, args):
        self.all_embeds_0 = tf.get_variable(
            "all_embeds_0",
            shape=[self.dataset.n_user + self.dataset.n_item, args.embed_size],
            initializer=self.initializer,
        )
        self.u_embeds_0, self.i_embeds_0 = tf.split(
            self.all_embeds_0, [self.dataset.n_user, self.dataset.n_item], 0
        )
        self.int_embeds_0, self.pop_embeds_0 = tf.split(self.all_embeds_0, 2, 1)

        embed_size = args.embed_size // 2
        if args.embed_type == "ngcf":
            self.int_W1s = []
            self.int_b1s = []
            self.int_W2s = []
            self.int_b2s = []
            self.pop_W1s = []
            self.pop_b1s = []
            self.pop_W2s = []
            self.pop_b2s = []
            for i in range(args.n_layer):
                self.int_W1s.append(
                    tf.get_variable(
                        "int_W1_{}".format(i),
                        shape=[embed_size, embed_size],
                        initializer=self.initializer,
                    )
                )
                self.int_b1s.append(
                    tf.get_variable(
                        "int_b1_{}".format(i),
                        shape=[1, embed_size],
                        initializer=self.initializer,
                    )
                )
                self.int_W2s.append(
                    tf.get_variable(
                        "int_W2_{}".format(i),
                        shape=[embed_size, embed_size],
                        initializer=self.initializer,
                    )
                )
                self.int_b2s.append(
                    tf.get_variable(
                        "int_b2_{}".format(i),
                        shape=[1, embed_size],
                        initializer=self.initializer,
                    )
                )
                self.pop_W1s.append(
                    tf.get_variable(
                        "pop_W1_{}".format(i),
                        shape=[embed_size, embed_size],
                        initializer=self.initializer,
                    )
                )
                self.pop_b1s.append(
                    tf.get_variable(
                        "pop_b1_{}".format(i),
                        shape=[1, embed_size],
                        initializer=self.initializer,
                    )
                )
                self.pop_W2s.append(
                    tf.get_variable(
                        "pop_W2_{}".format(i),
                        shape=[embed_size, embed_size],
                        initializer=self.initializer,
                    )
                )
                self.pop_b2s.append(
                    tf.get_variable(
                        "pop_b2_{}".format(i),
                        shape=[1, embed_size],
                        initializer=self.initializer,
                    )
                )

        if args.inference_type == "mlp":
            weight_sizes = [2 * embed_size] + [x // 2 for x in args.weight_sizes]
            self.int_Ws = []
            self.int_bs = []
            self.pop_Ws = []
            self.pop_bs = []
            for i in range(len(args.weight_sizes)):
                self.int_Ws.append(
                    tf.get_variable(
                        "int_W_{}".format(i),
                        shape=[weight_sizes[i], weight_sizes[i + 1]],
                        initializer=self.initializer,
                    )
                )
                self.int_bs.append(
                    tf.get_variable(
                        "int_b_{}".format(i),
                        shape=[1, weight_sizes[i + 1]],
                        initializer=self.initializer,
                    )
                )
            self.int_h = tf.get_variable(
                "int_h",
                shape=[weight_sizes[-1] + embed_size, 1],
                initializer=self.initializer,
            )
            for i in range(len(args.weight_sizes)):
                self.pop_Ws.append(
                    tf.get_variable(
                        "pop_W_{}".format(i),
                        shape=[weight_sizes[i], weight_sizes[i + 1]],
                        initializer=self.initializer,
                    )
                )
                self.pop_bs.append(
                    tf.get_variable(
                        "pop_b_{}".format(i),
                        shape=[1, weight_sizes[i + 1]],
                        initializer=self.initializer,
                    )
                )
            self.pop_h = tf.get_variable(
                "pop_h",
                shape=[weight_sizes[-1] + embed_size, 1],
                initializer=self.initializer,
            )

    def create_embeds(self, args):
        s_norm_adj = create_norm_adj(
            self.dataset.u_interacts,
            self.dataset.i_interacts,
            self.dataset.n_user,
            self.dataset.n_item,
        )
        if args.embed_type == "ngcf":
            self.int_embeds = create_ngcf_embed(
                self.int_embeds_0,
                s_norm_adj,
                args.n_layer,
                self.int_W1s,
                self.int_b1s,
                self.int_W2s,
                self.int_b2s,
                args,
            )
            self.pop_embeds = create_ngcf_embed(
                self.pop_embeds_0,
                s_norm_adj,
                args.n_layer,
                self.pop_W1s,
                self.pop_b1s,
                self.pop_W2s,
                self.pop_b2s,
                args,
            )
        elif args.embed_type == "lightgcn":
            self.int_embeds = create_lightgcn_embed(
                self.int_embeds_0, s_norm_adj, args.n_layer
            )
            self.pop_embeds = create_lightgcn_embed(
                self.pop_embeds_0, s_norm_adj, args.n_layer
            )
        self.all_embeds = tf.concat([self.int_embeds, self.pop_embeds], 1)
        self.u_embeds, self.i_embeds = tf.split(
            self.all_embeds, [self.dataset.n_user, self.dataset.n_item], 0
        )

    def create_loss(self, args):
        self.batch_pos_i = tf.placeholder(tf.int32, shape=(None,))
        self.batch_neg_i = tf.placeholder(tf.int32, shape=(None,))
        self.batch_neg_mask = tf.placeholder(tf.float32, shape=(None,))
        batch_pos_i_embeds = tf.nn.embedding_lookup(self.i_embeds, self.batch_pos_i)
        batch_neg_i_embeds = tf.nn.embedding_lookup(self.i_embeds, self.batch_neg_i)
        users_int, users_pop = tf.split(self.batch_u_embeds, 2, 1)
        items_p_int, items_p_pop = tf.split(batch_pos_i_embeds, 2, 1)
        items_n_int, items_n_pop = tf.split(batch_neg_i_embeds, 2, 1)
        if args.inference_type == "inner_product":
            p_score_int = inner_product(users_int, items_p_int)
            n_score_int = inner_product(users_int, items_n_int)
            p_score_pop = inner_product(users_pop, items_p_pop)
            n_score_pop = inner_product(users_pop, items_n_pop)
        elif args.inference_type == "mlp":
            p_score_int = mlp(
                users_int, items_p_int, self.int_Ws, self.int_bs, self.int_h, args
            )
            n_score_int = mlp(
                users_int, items_n_int, self.int_Ws, self.int_bs, self.int_h, args
            )
            p_score_pop = mlp(
                users_pop, items_p_pop, self.pop_Ws, self.pop_bs, self.pop_h, args
            )
            n_score_pop = mlp(
                users_pop, items_n_pop, self.pop_Ws, self.pop_bs, self.pop_h, args
            )

        p_score_total = p_score_int + p_score_pop
        n_score_total = n_score_int + n_score_pop

        self.loss_int = mask_bpr_loss(p_score_int, n_score_int, self.batch_neg_mask)
        self.loss_pop = mask_bpr_loss(
            n_score_pop, p_score_pop, self.batch_neg_mask
        ) + mask_bpr_loss(p_score_pop, n_score_pop, 1 - self.batch_neg_mask)
        self.loss_total = bpr_loss(p_score_total, n_score_total)

        user_int = tf.concat([users_int, users_int], 0)
        user_pop = tf.concat([users_pop, users_pop], 0)
        item_int = tf.concat([items_p_int, items_n_int], 0)
        item_pop = tf.concat([items_p_pop, items_n_pop], 0)
        self.discrepency_loss = discrepency_loss(item_int, item_pop) + discrepency_loss(
            user_int, user_pop
        )
        self.mf_loss = (
            self.int_weight * self.loss_int
            + self.pop_weight * self.loss_pop
            + self.loss_total
            - self.dis_pen * self.discrepency_loss
        )

        self.reg_loss = args.reg * l2_embed_loss(self.all_embeds)
        if args.embed_type == "ngcf":
            for x in (
                self.int_W1s
                + self.int_b1s
                + self.int_W2s
                + self.int_b2s
                + self.pop_W1s
                + self.pop_b1s
                + self.pop_W2s
                + self.pop_b2s
            ):
                self.reg_loss += args.weight_reg * tf.nn.l2_loss(x)
        if args.inference_type == "mlp":
            for x in (
                self.int_Ws
                + self.int_bs
                + [self.int_h]
                + self.pop_Ws
                + self.pop_bs
                + [self.pop_h]
            ):
                self.reg_loss += args.weight_reg * tf.nn.l2_loss(x)

        self.loss = self.mf_loss + self.reg_loss

    def create_batch_ratings(self, args):
        self.batch_u = tf.placeholder(tf.int32, shape=(None,))
        self.batch_u_embeds = tf.nn.embedding_lookup(self.u_embeds, self.batch_u)
        users_int, _ = tf.split(self.batch_u_embeds, 2, 1)
        i_int_embeds, _ = tf.split(self.i_embeds, 2, 1)

        if args.inference_type == "inner_product":
            self.batch_ratings = tf.matmul(users_int, i_int_embeds, transpose_b=True)
        elif args.inference_type == "mlp":
            u_size = tf.shape(users_int)[0]
            i_size = tf.shape(i_int_embeds)[0]
            u_repeats = tf.repeat(users_int, i_size, axis=0)
            i_tiles = tf.tile(i_int_embeds, [u_size, 1])
            scores = mlp(u_repeats, i_tiles, self.int_Ws, self.int_bs, self.int_h, args)
            self.batch_ratings = tf.reshape(scores, [u_size, i_size])

    def train_1_epoch(self, epoch, args):
        self.timer.start("Epoch {}".format(epoch))

        losses = []
        mf_losses = []
        dis_losses = []
        reg_losses = []
        for users, pos_items, neg_items, neg_mask in self.sampler.sample():
            (
                _,
                batch_loss,
                batch_mf_loss,
                batch_dis_loss,
                batch_reg_loss,
            ) = self.sess.run(
                [
                    self.opt,
                    self.loss,
                    self.mf_loss,
                    self.discrepency_loss,
                    self.reg_loss,
                ],
                feed_dict={
                    self.int_weight: self.int_weight_v,
                    self.pop_weight: self.pop_weight_v,
                    self.batch_u: users,
                    self.batch_pos_i: pos_items,
                    self.batch_neg_i: neg_items,
                    self.batch_neg_mask: neg_mask,
                },
            )
            losses.append(batch_loss)
            mf_losses.append(batch_mf_loss)
            dis_losses.append(batch_dis_loss)
            reg_losses.append(batch_reg_loss)

        print(
            "int_weight = {:.5f}, pop_weight = {:.5f}, margin = {:.5f}".format(
                self.int_weight_v, self.pop_weight_v, self.sampler.margin
            )
        )
        self.timer.stop(
            "loss = {:.5f} = {:.5f} (dis_loss = {:.5f}) + {:.5f}".format(
                np.mean(losses),
                np.mean(mf_losses),
                np.mean(dis_losses),
                np.mean(reg_losses),
            )
        )

    def eval(self, args):
        self.timer.start("Evaluation")
        for evaluator in self.evaluators.values():
            for idx, batch_u in enumerate(
                batch_iterator(evaluator.eval_users, args.eval_batch_size)
            ):
                batch_users_idx = range(
                    idx * args.eval_batch_size,
                    idx * args.eval_batch_size + len(batch_u),
                )
                batch_ratings = self.sess.run(
                    self.batch_ratings, feed_dict={self.batch_u: batch_u}
                )

                for idx, user in enumerate(batch_u):
                    batch_ratings[idx][self.dataset.train[user]] = -np.inf

                evaluator.update(batch_ratings, batch_users_idx)

            evaluator.update_final()
            print(evaluator)

        self.timer.stop()
