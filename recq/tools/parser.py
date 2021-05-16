import argparse


def parse_args(kind="bpr"):
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu_id",
                        type=int,
                        default=None,
                        help="ID of visible GPU.")
    parser.add_argument("--verbose_name",
                        default="",
                        help="Name to save the model.")
    parser.add_argument("--dataset", default="ifashion", help="Dataset.")
    parser.add_argument("--n_thread",
                        type=int,
                        default=10,
                        help="Number of threads.")
    parser.add_argument("--n_layer",
                        type=int,
                        default=3,
                        help="Number of convolutional layers.")
    parser.add_argument("--embed_size",
                        type=int,
                        default=128,
                        help="Embedding size.")
    parser.add_argument(
        "--n_i_group",
        type=int,
        default=1,
        help=
        "Number of item degree groups. Metrics will be calculated for all items and then for each group if n_i_group > 1."
    )
    parser.add_argument(
        "--n_u_group",
        type=int,
        default=1,
        help=
        "Number of user degree groups. Metrics will be calculated for all users and then for each group if n_u_group > 1."
    )
    parser.add_argument("--weight_sizes",
                        nargs="+",
                        type=int,
                        default=[],
                        help="Output dimensions of each MLP layer.")
    parser.add_argument("--lr",
                        type=float,
                        default=0.001,
                        help="Learning rate.")
    parser.add_argument("--reg",
                        type=float,
                        default=0.001,
                        help="Coefficient of regularization of embeddings.")
    parser.add_argument("--weight_reg",
                        type=float,
                        default=0.001,
                        help="Coefficient of regularization of weights.")
    parser.add_argument("--epoch",
                        type=int,
                        default=1000,
                        help="Number of epochs.")
    parser.add_argument("--batch_size",
                        type=int,
                        default=2048,
                        help="Batch size.")
    parser.add_argument("--eval_batch_size",
                        type=int,
                        default=2048,
                        help="Batch size for evaluation.")
    parser.add_argument("--inference_type",
                        default="inner_product",
                        help="Method to infer scores.")
    parser.add_argument("--print_info",
                        nargs="+",
                        default=[],
                        help="Which types of info to print.")
    parser.add_argument("--embed_type",
                        default="normalized",
                        help="Type of final embedding.")
    parser.add_argument("--train_type",
                        default="train",
                        help="Dataset for training.")
    parser.add_argument("--eval_types",
                        nargs="+",
                        default=["valid", "test"],
                        help="Datasets for evaluation.")
    parser.add_argument(
        "--eval_epoch",
        type=int,
        default=None,
        help=
        "If not None, evaluate the model every eval_epoch, else evaluation won't be performed."
    )
    parser.add_argument(
        "--early_stop",
        type=int,
        default=None,
        help=
        "If not None, stop training when best recall is decreasing for early_stop * eval_epoch\
        epochs, else early stopping won't be performed.")
    parser.add_argument("--metrics",
                        nargs="+",
                        default=["Recall", "Precision", "NDCG", "Rec", "ARP"],
                        help="Metrics calculated in groups.")
    parser.add_argument("--ks",
                        nargs="+",
                        type=int,
                        default=[10, 20, 50],
                        help="Top k(s) recommendation.")
    parser.add_argument("--save_model",
                        dest="save_model",
                        action="store_true",
                        help="Store the model after `epoch`.")
    parser.add_argument("--load_model",
                        default=None,
                        help="Name of the model to be loaded.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    if kind == "cpr":
        parser.add_argument("--sample_rate", type=float, default=1)
        parser.add_argument("--sample_ratio", type=float, default=3)
        parser.add_argument("--max_k_interact", type=int, default=3)
    elif kind == "ubpr":
        parser.add_argument("--ps_pow", type=float, default=0.5)
        parser.add_argument("--clip", type=float, default=0)
    elif kind == "dice":
        parser.add_argument("--int_weight", type=float, default=0.1)
        parser.add_argument("--pop_weight", type=float, default=0.1)
        parser.add_argument("--dis_pen", type=float, default=0.01)
        parser.add_argument("--margin", type=float, default=40)
        parser.add_argument("--min_size", type=float, default=40)
        parser.add_argument("--loss_decay", type=float, default=0.9)
        parser.add_argument("--margin_decay", type=float, default=0.9)

    return parser.parse_args()
