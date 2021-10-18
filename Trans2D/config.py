import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--run_name', type=str, default='test', help='name of the experiment')
parser.add_argument('--dataset_name', type=str, default='random', choices=['random'], help='name of the dataset')
parser.add_argument('--max_seq_len', type=int, default=1000, help='maximum length of a sequence')
parser.add_argument('--hidden_size', type=int, default=16, help='hidden embedding size')
parser.add_argument('--output_size', type=int, default=1, help='output size')

parser.add_argument('--transformer_type', type=str, default='Attention2D', help='kind of transformer over the features', choices=['Attention2D', 'Attention', 'GRU', 'LSTM'])
parser.add_argument('--pooling', type=str, default='mean', help='kind of pooling to apply before/after the transformer', choices=['mean', 'sum', 'transformer', 'concat'])
parser.add_argument('--num_attention_heads', type=int, default=4, help='number of attention heads in an attention layer')
parser.add_argument('--num_hidden_layers', type=int, default=1, help='number of attention blocks')
parser.add_argument('--max_position_embeddings', type=int, default=50, help='maximum length of a sequence')
parser.add_argument('--dropout', type=float, default=0.3, help='dropout rate')

parser.add_argument('--learning_rate', type=float, default=1e-3, help='initial learning rate')
parser.add_argument('--weight_decay', type=float, default=1e-5, help='optimizer weight decay')
parser.add_argument('--max_epochs', type=int, default=5, help='number of total epochs to run')
parser.add_argument('--batch_size', type=int, default=32, help='number of samples in batch')

parser.add_argument('--num_workers', type=int, default=0, help='number of workers for dataloader')
parser.add_argument('--gpus', type=str, default='0', help='gpus parameter used for pytorch_lightning')
parser.add_argument('--seed', type=int, default=43, help='random seed')
