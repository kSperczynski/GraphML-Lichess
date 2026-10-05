import pandas as pd
import os
from argparse import ArgumentParser
import numpy as np
import json

GAMES_FOLDER_DEFAULT = "games"
DATA_FOLDER_DEFAULT = "data"
PLAYERS_FOLDER_DEFAULT = "players"


def main(args):
    dataset = {
        "username": [],
        "win_streak": [],
        "loss_streak": [],
        "num_games": [],
        "win_rate": [],
        "loss_rate": [],
        "draw_rate": []
    }

    for filename in os.listdir(args.players_folder):
        username = filename.split('.')[0]

        with open(f"{args.players_folder}/{username}.json") as file:
            try:
                player_stats = json.load(file)
            except Exception as e:
                print(f"Skipping {username}")
                print(e)

        num_games = player_stats['stat']['count']['all']

        if num_games == 0:
            continue

        dataset['username'].append(username)
        dataset['win_streak'].append(player_stats['stat']['resultStreak']['win']['max']['v'])
        dataset['loss_streak'].append(player_stats['stat']['resultStreak']['loss']['max']['v'])
        dataset['num_games'].append(num_games)
        dataset['win_rate'].append(player_stats['stat']['count']['win'] / num_games)
        dataset['loss_rate'].append(player_stats['stat']['count']['loss'] / num_games)
        dataset['draw_rate'].append(player_stats['stat']['count']['draw'] / num_games)

    df = pd.DataFrame(dataset)

    df.to_csv(f"{args.data_folder}/player_features.csv", index=False)



if __name__ == "__main__":
    parser = ArgumentParser(description="Scrape games args")

    # parser.add_argument("--uname")
    # parser.add_argument(
    #     "--num_games", 
    #     type=int, 
    #     default=NUM_GAMES_DEFAULT
    # )
    parser.add_argument(
        "--games_folder", 
        default=GAMES_FOLDER_DEFAULT
    )
    parser.add_argument(
        "--data_folder",
        default=DATA_FOLDER_DEFAULT
    )
    parser.add_argument(
        "--players_folder",
        default=PLAYERS_FOLDER_DEFAULT
    )
    # parser.add_argument(
    #     "--checkpoint_folder", 
    #     default=CHECKPOINT_FOLDER_DEFAULT
    # )
    # parser.add_argument(
    #     "--queue_checkpoint"
    # )
    # parser.add_argument(
    #     "--opponents_folder",
    #     default=OPPONENTS_FOLDER_DEFAULT
    # )
    # parser.add_argument(
    #     "--matchup_results_folder",
    #     default=MATCHUP_RESULTS_FOLDER_DEFAULT
    # )

    args = parser.parse_args()

    main(args)