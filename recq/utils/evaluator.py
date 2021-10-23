import numpy as np
from recq.cyutils.evaluator import CyEvaluator
from recq.cyutils.evaluator import cy_recommend
import scipy.stats


def create_evaluators(dataset, eval_types, metrics, ks, n_thread):
    evaluators = {}
    for eval_type in eval_types:
        evaluators[eval_type] = Evaluator(
            dataset, eval_type, metrics, ks, (eval_type == "valid"), n_thread
        )
    return evaluators


def recommend(ratings, k, n_thread=8):
    return cy_recommend(ratings, k, n_thread)


class Evaluator(CyEvaluator):
    def __init__(
        self,
        dataset,
        eval_type,
        metrics=["Recall"],
        ks=[20],
        group=True,
        n_thread=8,
    ):
        self.eval_type = eval_type
        self.metrics = metrics
        if not group:
            self.metrics = [m for m in self.metrics if m != "Rec"]
        self.ks = ks
        self.i_degrees = dataset.i_degrees

        evalset = dataset.evalsets[eval_type]
        self.eval_users = np.array(sorted(evalset.keys()))

        if group:
            self.n_i_group = dataset.n_i_group
            self.n_u_group = dataset.n_u_group
            self.u_groups = (
                dataset.u_groups[self.eval_users]
                if dataset.u_groups is not None
                else None
            )
            self.i_groups = dataset.i_groups

        else:
            self.n_i_group = 1
            self.n_u_group = 1
            self.u_groups = None
            self.i_groups = None

        # (n_user, n_group_metric, n_k, n_i_group)
        self.metric_values_all_u = np.empty(
            [
                len(self.eval_users),
                len(self.metrics),
                len(ks),
                1 if self.n_i_group == 1 else self.n_i_group + 1,
            ],
            dtype=np.float32,
        )

        evalset_list = [evalset[u] for u in self.eval_users]
        super().__init__(
            evalset_list,
            [m.encode("utf_8") for m in self.metrics],
            ks,
            n_thread,
            self.n_i_group,
            self.i_groups,
            self.i_degrees,
            self.metric_values_all_u,
        )

    def update(self, batch_ratings, batch_users_idx):
        self.eval(batch_ratings, np.asarray(batch_users_idx, dtype=np.int32))

    def update_top_k(self, batch_top_k, batch_users_idx):
        self.eval_top_k(batch_top_k, np.asarray(batch_users_idx, dtype=np.int32))

    def update_final(self):
        self.metric_values_group_i = np.nanmean(self.metric_values_all_u, axis=0)
        if self.n_u_group > 1:
            # (n_metric, n_k, n_user)
            u_values = self.metric_values_all_u[:, :, :, 0].transpose(1, 2, 0)

            self.metric_values_group_u = np.stack(
                [np.nanmean(u_values, axis=-1)]
                + [
                    np.nanmean(u_values[:, :, self.u_groups == i], axis=-1)
                    for i in range(self.n_u_group)
                ],
                axis=-1,
            )

    def get(self, metric, k, group=-1):
        try:
            k_idx = self.ks.index(k)
        except:
            raise ValueError("{} is not in ks.".format(k))
        try:
            m_idx = self.metrics.index(metric)
        except:
            raise ValueError("{} is not in metrics.".format(metric))
        # (n_metric, n_k, n_i_group)
        return self.metric_values_group_i[m_idx, k_idx, group + 1]

    def _prep_lines(self, metrics, values):
        lines = []
        for metric, result_m in zip(self.metrics, values):
            if metric not in metrics:
                continue
            for k, result_m_k in zip(self.ks, result_m):
                lines.append(
                    "{:<10}@{:<3}:".format(metric, k)
                    + (
                        "{:10.5f}".format(result_m_k)
                        if isinstance(result_m_k, np.float32)
                        else "{}".format(
                            " ".join("{:10.5f}".format(x) for x in result_m_k)
                        )
                    )
                )
        return lines

    def __str__(self):
        lines = []
        lines.append("[ {} set ]".format(self.eval_type))
        lines.append("---- Item ----")
        lines.extend(self._prep_lines(self.metrics, self.metric_values_group_i))
        if self.n_u_group > 1:
            lines.append("---- User ----")
            lines.extend(
                self._prep_lines(
                    [m for m in self.metrics if m != "Rec"], self.metric_values_group_u
                )
            )
        return "\n".join(lines)
