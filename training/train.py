import torch
import torch_geometric
import argparse

def main():
    parser = argparse.ArgumentParser(description="Train predictor.")
    parser.add_argument("--num-epochs", type=int, default=10, required=False)
    parser.add_argument("--batch-size", type=int, default=32, required=False)
    parser.add_argument("--device", type=str, default='cuda', required=False)
    args = parser.parse_args()

    assert (not args.device=='cuda') or (torch.cuda.is_available()), 'ERR: attempting training on CUDA, missing CUDA device.'

if __name__ == "__main__":
    main()