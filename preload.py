import argparse


def preload(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--ad-no-huggingface",
        action="store_true",
        help="Don't use adetailer models from huggingface",
    )
