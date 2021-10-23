import tensorflow as tf
import numpy as np
from recq.recommenders.bpr import BPR
from recq.utils.data import CPRSampler
from recq.utils.inference import inner_product, mlp
from recq.utils.loss import cpr_loss


class CPR(BPR):
    def create_sampler(self, dataset, args):
        self.sampler = CPRSampler(dataset, args)

    def create_mf_loss(self, args):
        self.batch_pos_i = tf.placeholder(tf.int32, shape=(None,))
        batch_pos_i_embeds = tf.nn.embedding_lookup(self.i_embeds, self.batch_pos_i)
        pos_scores = []
        neg_scores = []
        # u_embeds: u1, u2, u1, u2, u3, ...
        u_splits = tf.split(
            self.batch_u_embeds, self.sampler.batch_total_sample_sizes, 0
        )
        i_splits = tf.split(
            batch_pos_i_embeds, self.sampler.batch_total_sample_sizes, 0
        )
        for idx in range(len(self.sampler.batch_total_sample_sizes)):
            u_list = tf.split(u_splits[idx], idx + 2, 0)
            i_list = tf.split(i_splits[idx], idx + 2, 0)
            if args.inference_type == "inner_product":
                pos_scores.append(
                    tf.reduce_mean(
                        [inner_product(u, i) for u, i in zip(u_list, i_list)], axis=0
                    )
                )
                neg_scores.append(
                    tf.reduce_mean(
                        [
                            inner_product(u, i)
                            for u, i in zip(u_list, i_list[1:] + [i_list[0]])
                        ],
                        axis=0,
                    )
                )
            elif args.inference_type == "mlp":
                pos_scores.append(
                    tf.reduce_mean(
                        [
                            mlp(u, i, self.Ws, self.bs, self.h, args)
                            for u, i in zip(u_list, i_list)
                        ],
                        axis=0,
                    )
                )
                neg_scores.append(
                    tf.reduce_mean(
                        [
                            mlp(u, i, self.Ws, self.bs, self.h, args)
                            for u, i in zip(u_list, i_list[1:] + [i_list[0]])
                        ],
                        axis=0,
                    )
                )
        pos_scores = tf.concat(pos_scores, axis=0)
        neg_scores = tf.concat(neg_scores, axis=0)

        self.mf_loss = cpr_loss(pos_scores, neg_scores, args)

    def train_1_epoch(self, epoch):
        self.timer.start("Epoch {}".format(epoch))
        losses = []
        mf_losses = []
        reg_losses = []
        for users, items in self.sampler.sample():
            _, batch_loss, batch_mf_loss, batch_reg_loss = self.sess.run(
                [self.opt, self.loss, self.mf_loss, self.reg_loss],
                feed_dict={self.batch_u: users, self.batch_pos_i: items},
            )
            losses.append(batch_loss)
            mf_losses.append(batch_mf_loss)
            reg_losses.append(batch_reg_loss)
        self.timer.stop(
            "loss = {:.5f} = {:.5f} + {:.5f}".format(
                np.mean(losses), np.mean(mf_losses), np.mean(reg_losses)
            )
        )
