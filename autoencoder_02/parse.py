import argparse

def parse_args():
     parser = argparse.ArgumentParser(description="Go Autoencoder")
     parser.add_argument('--batch_size', type=int, default=32,
                        help="the batch size for bpr loss training procedure")
     parser.add_argument('--epochs', type=int, default=2)
     parser.add_argument('--encoding_dim_input', type=int, default=784)
     parser.add_argument('--encoding_dim_output', type=int, default=784)
     return parser.parse_args()
