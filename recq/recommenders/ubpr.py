import tensorflow as tf
import numpy as np
from recq.recommenders.bpr import BPR
from recq.utils.data import UBPRSampler
from recq.utils.loss import ubpr_loss


class UBPR(BPR):
    def create_sampler(self, dataset, args):
        self.sampler = UBPRSampler(dataset, args)

    def create_mf_loss(self, args):
        self.batch_i = tf.placeholder(tf.int32, shape=(None,))
        self.batch_j = tf.placeholder(tf.int32, shape=(None,))
        self.batch_j_labels = tf.placeholder(tf.float32, shape=(None,))
        batch_i_embeds = tf.nn.embedding_lookup(self.i_embeds, self.batch_i)
        batch_j_embeds = tf.nn.embedding_lookup(self.i_embeds, self.batch_j)
        self.p_scores = np.power(
            self.dataset.i_degrees / self.dataset.i_degrees.max(),
            args.ps_pow,
            dtype=np.float32,
        )
        batch_i_p_scores = tf.nn.embedding_lookup(self.p_scores, self.batch_i)
        batch_j_p_scores = tf.nn.embedding_lookup(self.p_scores, self.batch_j)
        self.mf_loss = ubpr_loss(
            self.batch_u_embeds,
            batch_i_embeds,
            batch_j_embeds,
            batch_i_p_scores,
            batch_j_p_scores,
            self.batch_j_labels,
            args,
        )

    def train_1_epoch(self, epoch):
        self.timer.start("Epoch {}".format(epoch))
        losses = []
        mf_losses = []
        reg_losses = []
        for users, i_items, j_items, j_labels in self.sampler.sample():
            _, batch_loss, batch_mf_loss, batch_reg_loss = self.sess.run(
                [self.opt, self.loss, self.mf_loss, self.reg_loss],
                feed_dict={
                    self.batch_u: users,
                    self.batch_i: i_items,
                    self.batch_j: j_items,
                    self.batch_j_labels: j_labels,
                },
            )
            losses.append(batch_loss)
            mf_losses.append(batch_mf_loss)
            reg_losses.append(batch_reg_loss)
        self.timer.stop(
            "loss = {:.5f} = {:.5f} + {:.5f}".format(
                np.mean(losses), np.mean(mf_losses), np.mean(reg_losses)
            )
        )
