import random
import numpy as np
from math import ceil
from recq.tools.monitor import Timer
from recq.cyutils.sampler import CyCPRSampler, CyPairNegSampler, CyDICENegSampler
from recq.cyutils.labeler import CyLabeler


def batch_iterator(data, batch_size, drop_last=False):
    """Generate batches.

    Args:
        data (list or numpy.ndarray): Input data.
        batch_size (int): Size of each batch except for the last one.
    """
    length = len(data)
    if drop_last:
        n_batch = length // batch_size
    else:
        n_batch = ceil(length / batch_size)
    for i in range(n_batch):
        yield data[i * batch_size : (i + 1) * batch_size]


class BPRSampler(object):
    def __init__(self, dataset, args):
        random.seed(args.seed)
        np.random.seed(args.seed)
        self.batch_size = args.batch_size
        self.n_thread = args.n_thread
        self.args = args

        self.n_user = dataset.n_user
        self.n_item = dataset.n_item
        self.n_interact = dataset.n_interact
        self.train = dataset.train
        self.train_inverse = dataset.train_inverse
        self.u_interacts = dataset.u_interacts
        self.i_interacts = dataset.i_interacts

        self.n_step = dataset.n_interact // self.batch_size
        self.set_samplers(self.args)

    def set_samplers(self, args):
        self.batch_sample_size = self.batch_size
        self.sample_size = self.n_step * self.batch_sample_size
        self.choice_size = 2 * self.sample_size
        self.neg_items = np.empty(self.sample_size, dtype=np.int32)
        self.neg_i_sampler = CyPairNegSampler(
            self.train,
            self.neg_items,
            self.n_step,
            self.batch_sample_size,
            args.n_thread,
        )

    def sample(self):
        idx = np.random.choice(self.n_interact, size=self.sample_size)
        users = self.u_interacts[idx]
        pos_items = self.i_interacts[idx]
        rand_items = np.random.choice(self.n_item, size=self.choice_size).astype(
            np.int32
        )
        if self.neg_i_sampler.sample(users, rand_items) == -1:
            raise RuntimeError("choice_size is not large enough")
        return zip(
            batch_iterator(users, batch_size=self.batch_sample_size),
            batch_iterator(pos_items, batch_size=self.batch_sample_size),
            batch_iterator(self.neg_items, batch_size=self.batch_sample_size),
        )


class UBPRSampler(BPRSampler):
    def set_samplers(self, args):
        self.batch_sample_size = self.batch_size
        self.sample_size = self.n_step * self.batch_sample_size
        self.choice_size = 2 * self.sample_size
        self.j_labels = np.empty(self.sample_size, dtype=np.float32)
        self.labeler = CyLabeler(
            self.train,
            self.j_labels,
            self.n_step,
            self.batch_sample_size,
            args.n_thread,
        )

    def sample(self):
        idx = np.random.choice(self.n_interact, size=self.sample_size)
        users = self.u_interacts[idx]
        i_items = self.i_interacts[idx]
        j_items = np.random.choice(self.n_item, size=self.sample_size).astype(np.int32)
        self.labeler.label(users, j_items)
        return zip(
            batch_iterator(users, batch_size=self.batch_sample_size),
            batch_iterator(i_items, batch_size=self.batch_sample_size),
            batch_iterator(j_items, batch_size=self.batch_sample_size),
            batch_iterator(self.j_labels, batch_size=self.batch_sample_size),
        )


class CPRSampler(BPRSampler):
    def set_samplers(self, args):
        if args.k_interact is None:
            ratios = np.power(
                args.sample_ratio, np.arange(args.max_k_interact - 2, -1, -1)
            )
        else:
            ratios = np.array([0] * (args.k_interact - 2) + [1])
        batch_sizes = np.round(args.batch_size / np.sum(ratios) * ratios).astype(
            np.int32
        )
        batch_sizes[-1] = args.batch_size - np.sum(batch_sizes[:-1])
        self.batch_sample_sizes = np.ceil(
            np.array(batch_sizes) * args.sample_rate
        ).astype(np.int32)
        self.batch_total_sample_sizes = self.batch_sample_sizes * np.arange(
            2, len(self.batch_sample_sizes) + 2
        )
        self.batch_sample_size = np.sum(self.batch_total_sample_sizes)
        self.sample_size = self.n_step * self.batch_sample_size
        self.batch_choice_sizes = 2 * self.batch_sample_sizes
        self.choice_size = 2 * self.sample_size

        self.users = np.empty(self.sample_size, dtype=np.int32)
        self.items = np.empty(self.sample_size, dtype=np.int32)
        self.cpr_sampler = CyCPRSampler(
            self.train,
            self.u_interacts,
            self.i_interacts,
            self.users,
            self.items,
            self.n_step,
            self.batch_sample_sizes,
            args.n_thread,
        )

    def sample(self):
        interact_idx = np.random.choice(self.n_interact, size=self.choice_size).astype(
            np.int32
        )

        if self.cpr_sampler.sample(interact_idx, self.batch_choice_sizes) == -1:
            raise RuntimeError("choice_size is not large enough")

        return zip(
            batch_iterator(self.users, self.batch_sample_size),
            batch_iterator(self.items, self.batch_sample_size),
        )


class DICESampler(object):
    def __init__(self, dataset, args):
        random.seed(args.seed)
        np.random.seed(args.seed)
        self.batch_size = args.batch_size
        self.n_thread = args.n_thread

        self.n_user = dataset.n_user
        self.n_item = dataset.n_item
        self.n_interact = dataset.n_interact
        self.train = dataset.train
        self.u_interacts = dataset.u_interacts
        self.i_interacts = dataset.i_interacts
        self.i_degrees = dataset.i_degrees
        self.n_step = dataset.n_interact // self.batch_size
        self.sample_size = self.n_step * self.batch_size

        self.margin = args.margin
        self.min_size = args.min_size

        self.neg_items = np.empty(self.sample_size, dtype=np.int32)
        self.neg_mask = np.empty(self.sample_size, dtype=np.float32)
        self.sampler = CyDICENegSampler(
            self.train,
            self.n_item,
            self.min_size,
            self.i_degrees,
            self.neg_items,
            self.neg_mask,
            self.n_thread,
        )

    def sample(self):
        idx = np.random.choice(self.n_interact, size=self.sample_size)
        users = self.u_interacts[idx]
        pos_items = self.i_interacts[idx]
        rand = np.random.rand(self.sample_size * 2).astype(np.float32)
        self.sampler.sample(users, pos_items, rand, self.margin)
        return zip(
            batch_iterator(users, batch_size=self.batch_size),
            batch_iterator(pos_items, batch_size=self.batch_size),
            batch_iterator(self.neg_items, batch_size=self.batch_size),
            batch_iterator(self.neg_mask, batch_size=self.batch_size),
        )
