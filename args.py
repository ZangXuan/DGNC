import argparse


dataset = "primaryschool"


def forum_params():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default="fb-forum-pre")
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--node_clf', type=bool, default=False)
    parser.add_argument('--out_dim', type=int, default=64)
    parser.add_argument('--num_nodes', type=int, default=900)
    parser.add_argument('--train_span', type=int, default=8, help='train_span<11')

    # The parameters of learning process
    parser.add_argument('--epoch', type=int, default=300)
    parser.add_argument('--lr', type=float, default=0.005)
    parser.add_argument('--early_stop', type=int, default=200, help='patient')

    # model-specific parameters
    parser.add_argument('--t_span', type=int, default=1296000)
    parser.add_argument('--feat_drop', type=float, default=0.1)
    parser.add_argument('--attn_drop', type=float, default=0.5)
    parser.add_argument('--K', type=int, default=12)
    parser.add_argument('--tau', type=float, default=0.5)
    parser.add_argument('--lam', type=float, default=0.1)

    args, _ = parser.parse_known_args()
    return args

def enron_params():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default="enron")
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--node_clf', type=bool, default=False)
    parser.add_argument('--out_dim', type=int, default=64)
    parser.add_argument('--num_nodes', type=int, default=151)
    parser.add_argument('--train_span', type=int, default=9, help='train_span<10')
    # parser.add_argument('--batch_size', type=int, default=128)

    # The parameters of learning process
    parser.add_argument('--epoch', type=int, default=300)
    parser.add_argument('--lr', type=float, default=0.01)
    # parser.add_argument('--early_stop', type=int, default=200, help='patient')
    parser.add_argument('--early_stop', type=int, default=50, help='patient')

    # model-specific parameterse
    parser.add_argument('--t_span', type=int, default=9828449)
    parser.add_argument('--feat_drop', type=float, default=0.1)
    parser.add_argument('--attn_drop', type=float, default=0.5)
    parser.add_argument('--K', type=int, default=24)
    parser.add_argument('--tau', type=float, default=0.5)
    parser.add_argument('--lam', type=float, default=0.2)

    args, _ = parser.parse_known_args()
    return args

def UCI_params():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default="UCI")
    parser.add_argument('--gpu', type=int, default=1)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--node_clf', type=bool, default=False)
    parser.add_argument('--out_dim', type=int, default=64)
    parser.add_argument('--num_nodes', type=int, default=1900)
    parser.add_argument('--train_span', type=int, default=12, help='train_span<13')
    # parser.add_argument('--batch_size', type=int, default=128)

    # The parameters of learning process
    parser.add_argument('--epoch', type=int, default=200)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--early_stop', type=int, default=100, help='patient')

    # model-specific parameters
    parser.add_argument('--t_span', type=int, default=1287400)
    parser.add_argument('--feat_drop', type=float, default=0.1)
    parser.add_argument('--attn_drop', type=float, default=0.5)
    # parser.add_argument('--K', type=int, default=20)
    parser.add_argument('--K', type=int, default=8)
    parser.add_argument('--tau', type=float, default=0.5)
    parser.add_argument('--lam', type=float, default=0.3)

    args, _ = parser.parse_known_args()
    return args

def primaryschool_params():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default="primaryschool")
    parser.add_argument('--gpu', type=int, default=2)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--node_clf', type=bool, default=True)
    parser.add_argument('--out_dim', type=int, default=64)
    parser.add_argument('--num_nodes', type=int, default=242)
    parser.add_argument('--train_span', type=int, default=9, help='train_span<10')
    # parser.add_argument('--batch_size', type=int, default=128)

    # The parameters of learning process
    parser.add_argument('--epoch', type=int, default=300)
    parser.add_argument('--lr', type=float, default=0.008)
    parser.add_argument('--early_stop', type=int, default=200, help='patient')

    # model-specific parameters
    parser.add_argument('--t_span', type=int, default=3200)
    parser.add_argument('--feat_drop', type=float, default=0.1)
    parser.add_argument('--attn_drop', type=float, default=0.5)
    parser.add_argument('--K', type=int, default=11)
    parser.add_argument('--tau', type=float, default=0.5)
    parser.add_argument('--lam', type=float, default=0.2)

    args, _ = parser.parse_known_args()
    return args


def set_params():
    if dataset == "fb-forum-pre":
        args = forum_params()
    elif dataset == "primaryschool":
        args = primaryschool_params()
    elif dataset == "UCI":
        args = UCI_params()
    elif dataset == "enron":
        args = enron_params()
    return args
