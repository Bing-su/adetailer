import argparse


def preload(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--ad-no-huggingface",
        action="store_true",
        help="Don't use adetailer models from huggingface",
    )
    parser.add_argument(
        "--adetailer-dir", 
        type=str, 
        help="directory with adetailer models", 
        default=None
    )
