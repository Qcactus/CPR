import numpy as np
import scipy.sparse
import tensorflow as tf
from recq.tools.monitor import Timer
from recq.utils.tf_utils import sp_mat_2_sp_tensor


def create_norm_adj(u_interacts, i_interacts, n_user, n_item):
    """Create normalized adjacency matrix.

    Returns:
        scipy.sparse.csr_matrix: Normalized adjacency matrix.

    """

    # Create interaction matrix.
    R = scipy.sparse.coo_matrix(
        ([1.0] * len(u_interacts), (u_interacts, i_interacts)),
        shape=(n_user, n_item),
        dtype=np.float32,
    ).tocsr()

    # Create adjacency matrix.
    zero_u_mat = scipy.sparse.csr_matrix((n_user, n_user), dtype=np.float32)
    zero_i_mat = scipy.sparse.csr_matrix((n_item, n_item), dtype=np.float32)
    adj = scipy.sparse.hstack(
        [scipy.sparse.vstack([zero_u_mat, R.T]), scipy.sparse.vstack([R, zero_i_mat])]
    ).tocsr()

    D = np.array(adj.sum(axis=1))
    # Normalize adjacency matrix.
    row_sum = D.ravel()
    # Symmetric normalized Laplacian
    s_diag_flat = np.power(
        row_sum, -0.5, out=np.zeros_like(row_sum), where=row_sum != 0
    )
    s_diag = scipy.sparse.diags(s_diag_flat)
    s_norm_adj = s_diag.dot(adj).dot(s_diag)

    return s_norm_adj


def create_ngcf_embed(all_embeds_0, s_norm_adj, n_layer, W1s, b1s, W2s, b2s, args):
    s_norm_adj = sp_mat_2_sp_tensor(s_norm_adj)
    ego_embeds = all_embeds_0
    all_embeds = [ego_embeds]

    for i in range(n_layer):
        neigh_embeds = tf.sparse_tensor_dense_matmul(s_norm_adj, ego_embeds)
        sum_embeds = tf.nn.leaky_relu(tf.matmul(neigh_embeds, W1s[i]) + b1s[i])
        bi_embeds = tf.nn.leaky_relu(
            tf.matmul(tf.multiply(ego_embeds, neigh_embeds), W2s[i]) + b2s[i]
        )
        ego_embeds = sum_embeds + bi_embeds
        all_embeds += [tf.nn.l2_normalize(ego_embeds, axis=1)]

    all_embeds = tf.concat(all_embeds, 1)

    return all_embeds


def create_lightgcn_embed(all_embeds_0, s_norm_adj, n_layer):
    s_norm_adj = sp_mat_2_sp_tensor(s_norm_adj)
    ego_embeds = all_embeds_0
    all_embeds = [ego_embeds]

    for _ in range(n_layer):
        ego_embeds = tf.sparse_tensor_dense_matmul(s_norm_adj, ego_embeds)
        all_embeds += [ego_embeds]

    all_embeds = tf.stack(all_embeds, 1)
    all_embeds = tf.reduce_mean(all_embeds, axis=1, keepdims=False)

    return all_embeds
