import pandas as pd
import os
from argparse import ArgumentParser
import numpy as np

DATA_FOLDER_DEFAULT = "data"

def main(args):
    games_df = pd.read_csv(f"{args.data_folder}/game_features.csv")
    players_df = pd.read_csv(f"{args.data_folder}/player_features.csv")

    merged = pd.merge(games_df, players_df, on="username", how="inner")

    merged.to_csv(f"{args.data_folder}/node_features.csv", index=False)


if __name__ == "__main__":
    parser = ArgumentParser(description="Scrape games args")

    parser.add_argument(
        "--data_folder",
        default=DATA_FOLDER_DEFAULT
    )

    args = parser.parse_args()

    main(args)