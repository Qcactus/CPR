import os
from pickle import load
import numpy as np
from recq.tools.io import load_csv_2dict
from recq.tools.dataformat import invert_dict
from recq.tools.monitor import Timer


def count_interacts(dataset):
    count = 0
    for x in dataset.values():
        count += len(x)
    return count


def print_dataset_info(dataset):
    inverse = invert_dict(dataset)
    n_user = len(dataset)
    n_item = len(inverse)
    n_interact = count_interacts(dataset)
    min_d_u = min(len(x) for x in dataset.values())
    max_d_u = max(len(x) for x in dataset.values())
    min_d_i = min(len(x) for x in inverse.values())
    max_d_i = max(len(x) for x in inverse.values())
    print(
        "Number of users =",
        n_user,
        ", Number of items =",
        n_item,
        ", Number of interactions =",
        n_interact,
    )
    print("Sparsity:", n_interact / (n_user * n_item))
    print("Min degree (U) =", min_d_u, ", Max degree (U) =", max_d_u)
    print("Min degree (I) =", min_d_i, ", Max degree (I) =", max_d_i)


def get_degrees(dataset, n_node):
    degrees = np.array(
        [len(dataset[u]) if u in dataset else 0 for u in range(n_node)], dtype=np.int32
    )
    return degrees


def group_by_degree(n_group, degrees):
    sum_degrees = degrees.sum()
    degree_sort = np.argsort(degrees)
    degree_cumsum = degrees.copy()
    cum_sum = 0
    for x in degree_sort:
        cum_sum += degree_cumsum[x]
        degree_cumsum[x] = cum_sum
    split_idx = np.linspace(0, sum_degrees, n_group + 1)
    groups = np.searchsorted(split_idx[1:-1], degree_cumsum).astype(np.int32)
    return groups


def print_group_info(n_group, groups, degrees):
    for i in range(n_group):
        group_i = degrees[groups == i]
        print(
            "Group {:2d}: Number of nodes = {:6d}, Sum of degrees = {:7d}, Min of degrees = {:4d}, Max of degrees = {:5d}".format(
                i, group_i.size, group_i.sum(), group_i.min(), group_i.max()
            )
        )


class Dataset(object):
    def __init__(self, args, data_dir):
        self.n_i_group = args.n_i_group
        self.n_u_group = args.n_u_group

        self.i_groups = None
        self.u_groups = None

        # Load train, valid, test set.
        self.train = load_csv_2dict(data_dir, args.train_type)
        self.train_inverse = invert_dict(self.train)
        self.evalsets = {}
        for name in args.eval_types:
            self.evalsets[name] = load_csv_2dict(data_dir, name)

        self.n_user = max(self.train.keys()) + 1
        self.n_item = max(self.train_inverse.keys()) + 1

        if "dataset" in args.print_info:
            print("train:")
            print_dataset_info(self.train)
            for name in args.eval_types:
                print(name + ":")
                print_dataset_info(self.evalsets[name])

        self.u_degrees = get_degrees(self.train, self.n_user)
        self.i_degrees = get_degrees(self.train_inverse, self.n_item)

        self.train = [
            self.train[u] if u in self.train else [] for u in range(self.n_user)
        ]
        self.train_inverse = [
            self.train_inverse[i] if i in self.train_inverse else []
            for i in range(self.n_item)
        ]
        self.u_interacts = []
        self.i_interacts = []
        for u, items in enumerate(self.train):
            for i in items:
                self.u_interacts.append(u)
                self.i_interacts.append(i)
        self.u_interacts = np.array(self.u_interacts, dtype=np.int32)
        self.i_interacts = np.array(self.i_interacts, dtype=np.int32)
        self.n_interact = self.u_interacts.shape[0]

        # Divide nodes into groups by degree.
        if self.n_u_group > 1:
            print("Group users by degrees in train...")
            self.u_groups = group_by_degree(self.n_u_group, self.u_degrees)
            if "group" in args.print_info:
                print("[ train set ]")
                print_group_info(self.n_u_group, self.u_groups, self.u_degrees)
                for name in args.eval_types:
                    u_degrees = get_degrees(self.evalsets[name], self.n_user)
                    print("[ {} set ]".format(name))
                    print_group_info(self.n_u_group, self.u_groups, u_degrees)
        if self.n_i_group > 1:
            print("Group items by degrees in train...")
            self.i_groups = group_by_degree(self.n_i_group, self.i_degrees)
            if "group" in args.print_info:
                print("[ train set ]")
                print_group_info(self.n_i_group, self.i_groups, self.i_degrees)
                for name in args.eval_types:
                    i_degrees = get_degrees(
                        invert_dict(self.evalsets[name]), self.n_item
                    )
                    print("[ {} set ]".format(name))
                    print_group_info(self.n_i_group, self.i_groups, i_degrees)
