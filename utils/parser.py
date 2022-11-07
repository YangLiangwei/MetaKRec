import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset', type=str, default='music_kg3,music_kg8,music_kg4',
                        help='which dataset to train(dataset: book, movie1m, music, restaurant, movie20m)')
    parser.add_argument('-b', '--batch_size', type=int, default=256, help='set the batch size of training')
    parser.add_argument('-t', '--tune', type=int, default=0, help='tune the parameters.')
    parser.add_argument('-m', '--model', type=str, default='metakrec',help='which module to choice(module: gcn, hgcn...)')
    parser.add_argument('-D', '--dim', type=int, default=4, help='embedding dimension')
    parser.add_argument('-h1', '--hidden1', type=int, default=64, help='the hidden_1 is u_hidden_size')
    parser.add_argument('-h2', '--hidden2', type=int, default=32, help='the hidden_2 is i_hidden_size')
    parser.add_argument('-Depth', '--num_layers', type=int, default=1, help="number of layers of hetergcn")

    parser.add_argument('-E', '--epochs', type=int, default=300, help='the epochs of the training')
    parser.add_argument('-l', '--lr', type=float, default=0.1, help='learning rate') # 1e-4
    parser.add_argument('-w', '--l2_weight_decay', type=float, default=0.001, help='l2_weight_decay')
    parser.add_argument('-e', '--early_stop_patience', type=int, default=5, help='the parameter for stopping early')
    parser.add_argument('-topk', '--show_topk', type=int, default=0, help="1:show topk;0:don't show topk")
    parser.add_argument('-time', '--show_time', type=int, default=1, help="1:show time;0:don't show time")
    parser.add_argument('--clamp', type=int, default=0, help='1:using clamp;0: softplus')
    parser.add_argument('--coldstartexp', type=bool, default=False, help='True:cold start experiment;False:normal experiment')
    parser.add_argument('--gpu', type = int, default = 0, help = "which gpu to use, -1 indicates uses cpu")

    parser.add_argument('--type', type = str, default = 'attention', help = "channel aggregation type")

    return parser.parse_args()
