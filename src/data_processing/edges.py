import pandas as pd
import os
from argparse import ArgumentParser
import numpy as np

DATA_FOLDER_DEFAULT = "data"
H2H_FOLDER_DEFAULT = "head-to-head-results"

def main(args):
    edges = []
    for filename in os.listdir(args.h2h):
        edges.append(pd.read_csv(f"{args.h2h}/{filename}"))

    nodes = pd.read_csv(f"{args.data_folder}/node_features.csv")
    edges = pd.concat(edges)

    valid = set(nodes["username"])

    from_ok = edges["from"].isin(valid)
    to_ok = edges["to"].isin(valid)

    edges_clean = edges[from_ok & to_ok]
    edges_clean.to_csv(f"{args.data_folder}/edges.csv", index=False)

    missing = sorted(set(edges.loc[~from_ok, "from"].dropna()) | set(edges.loc[~to_ok, "to"].dropna()))
    pd.Series(missing, name="username").to_csv(f"{args.data_folder}/missing_usernames.csv", index=False)

    print(f"Kept {len(edges_clean)} of {len(edges)} edges")
    print(f"{len(missing)} usernames appear in edges but not in nodes")

if __name__ == "__main__":
    parser = ArgumentParser(description="Scrape games args")

    parser.add_argument(
        "--data_folder",
        default=DATA_FOLDER_DEFAULT
    )

    parser.add_argument(
        "--h2h",
        default=H2H_FOLDER_DEFAULT
    )

    args = parser.parse_args()

    main(args)