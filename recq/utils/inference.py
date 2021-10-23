import tensorflow as tf


def inner_product(u_embeds, i_embeds):
    output = tf.reduce_sum(u_embeds * i_embeds, axis=1)
    return output


def mlp(u_embeds, i_embeds, Ws, bs, h, args):
    mf_output = u_embeds * i_embeds
    mlp_output = tf.concat([u_embeds, i_embeds], axis=1)
    for i in range(len(args.weight_sizes)):
        mlp_output = tf.nn.relu(tf.matmul(mlp_output, Ws[i]) + bs[i])
    output = tf.reshape(tf.matmul(tf.concat([mf_output, mlp_output], axis=1), h), [-1])
    return output
