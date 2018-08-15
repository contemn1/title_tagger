import argparse


def init_argument_parser():
    parser = argparse.ArgumentParser(description="PyTorch MNIST Example")
    parser.add_argument("--training-dir", type=str, metavar="N",
                        help="path of training directory")

    parser.add_argument("--training-file", type=str, metavar="N",
                        help="name of training file")

    parser.add_argument("--batch-size", type=int, default=64, metavar="N",
                        help="input batch size for training (default: 64)")

    parser.add_argument("--word-vec-size", type=int, default=300, metavar="N",
                        help="dimension of word embedding")

    parser.add_argument("--enc-layers", type=int, default=1, metavar="N",
                        help="number of layers of encoder RNNs")

    parser.add_argument("--dec-layers", type=int, default=1, metavar="N",
                        help="number of layers of decoder RNNs")

    parser.add_argument("--dropout", type=float, default=0.5, metavar="N",
                        help="dropout rate")

    parser.add_argument("--model-path", type=str, metavar="N",
                        help="path to save model")

    parser.add_argument("--rnn-size", type=int, default=300, metavar="N",
                        help="hidden size of RNN cell")

    parser.add_argument("--bidirectional", type=bool, default=True, metavar="N",
                        help="whether to use bidirectional RNN")

    parser.add_argument("--attention-mode", type=str, metavar="N",
                        default="general",
                        help="type of attention general, dot or concat")

    parser.add_argument("--input-feeding", type=bool, default=False,
                        metavar="N",
                        help="whether to use input feeding")

    parser.add_argument("--train-ml", type=bool, default=True,
                        metavar="N",
                        help="whether to use maximum log likelihood in loss function")

    parser.add_argument("--train-rl", type=bool, default=False,
                        metavar="N",
                        help="whether to use self crictic in loss function")

    parser.add_argument("--rl-rate", type=float, default=0.9,
                        help="factor of reinforcement learning")
    parser.add_argument("--learning-rate", type=float, default=0.001,
                        metavar="N",
                        help="learning rate for maximum likelihood")

    parser.add_argument("--learning-rate-rl", type=float, default=0.001,
                        metavar="N",
                        help="learning rate for reinforcement learning")

    parser.add_argument("--max-grad-norm", type=float, default=4.0, metavar="N",
                        help="dropout rate")

    parser.add_argument("--num-epoches", type=int, default=100, metavar="N",
                        help="number of epoches")

    parser.add_argument("--start-epoch", type=int, default=1, metavar="N",
                        help="number of start epoches")

    parser.add_argument("--run-valid-every", type=int, default=10000,
                        metavar="N",
                        help="number of epochs to run validation set")

    parser.add_argument("--save-model-every", type=int, default=20000,
                        metavar="N",
                        help="number of epochs to run validation set")

    parser.add_argument("--early-stop-tolerance", type=int, default=20,
                        metavar="N",
                        help="number of epochs to run validation set")

    parser.add_argument("--min-word-freq", type=int, default=10,
                        metavar="N", help="minimum word frequency")

    parser.add_argument("--min-tag-freq", type=int, default=10,
                        metavar="N", help="minimum word frequency")

    parser.add_argument("--dist-backend", default="nccl", type=str,
                        help="distributed backend")

    parser.add_argument("--print-loss-every", type=int, default=50)

    parser.add_argument("--restore-model", type=bool, default=False)

    parser.add_argument("--model-name", type=str, help="name of saved model")

    parser.add_argument("--word-index-map-name", type=str, help="name of word "
                                                                "index map")
    parser.add_argument("--tag-index-map-name", type=str,
                        help="name of tag index map")

    parser.add_argument("--normalize-attention", type=bool, default=False,
                        help="whether to normalize encoder decoder attention")
    parser.add_argument("--store-dict", type=bool, default=True,
                        help="whether to store word index map")
    parser.add_argument("--previous-output-dir", type=str,
                        help="previous job output directory")

    return parser.parse_args()
