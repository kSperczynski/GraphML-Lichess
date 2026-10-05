import pandas as pd
import os
from argparse import ArgumentParser
import numpy as np

GAMES_FOLDER_DEFAULT = "games"
DATA_FOLDER_DEFAULT = "data"

def main(args):
    dataset = []
    for filename in os.listdir(args.games_folder):
        username = filename.split('.')[0]

        games_df = parse_games(username, args.games_folder)
        if games_df is None:
            continue
        if len(games_df) < 10:
            continue
        aggregated_game_stats = calculate_aggregate_game_stats(games_df)

        # print(aggregated_game_stats)

        dataset.append(aggregated_game_stats)

    dataset_df = pd.DataFrame(dataset)

    dataset_df.to_csv(f"{args.data_folder}/game_features.csv", index=False)

def parse_games(username, folder):
    df = pd.read_json(f"{folder}/{username}.json")

    players_df = pd.json_normalize(df["players"])
    opening_df = pd.json_normalize(df['opening'])
    clock_df = pd.json_normalize(df['clock'])
    
    df = df.drop(columns="players").join(players_df)
    df = df.drop(columns="opening").join(opening_df)
    df = df.drop(columns="clock").join(clock_df)

    print(df.columns)

    if 'clocks' not in df.columns:
        print(f"NO CLOCKS; SKIPPING {username}")
        return None

    df["avg_move_time"] = df.apply(avg_move_time, axis=1, args=(username,))

    print(df.columns)
        
    white_games = df[df["white.user.name"] == username][["createdAt", "white.rating", "white.analysis.acpl", "white.analysis.accuracy", "white.analysis.inaccuracy", "white.analysis.mistake", "white.analysis.blunder", "moves", "ply", "avg_move_time"]]
    white_games.rename(columns={"white.rating": "rating", "white.analysis.acpl": "acpl", "white.analysis.accuracy": "accuracy", "white.analysis.inaccuracy": "num_inaccuracies", "white.analysis.mistake": "num_mistakes", "white.analysis.blunder": "num_blunders", "ply": "last_theory_ply"}, inplace=True)
    
    black_games = df[df["black.user.name"] == username][["createdAt", "black.rating", "black.analysis.acpl", "black.analysis.accuracy", "black.analysis.inaccuracy", "black.analysis.mistake", "black.analysis.blunder", "moves", "ply", "avg_move_time"]]
    black_games.rename(columns={"black.rating": "rating", "black.analysis.acpl": "acpl", "black.analysis.accuracy": "accuracy", "black.analysis.inaccuracy": "num_inaccuracies", "black.analysis.mistake": "num_mistakes", "black.analysis.blunder": "num_blunders", "ply": "last_theory_ply"}, inplace=True)
    
    res = pd.concat([white_games, black_games])
    
    res["num_moves"] = res["moves"].str.split(" ").str.len()
    res["frac_blunders"] = res["num_blunders"] / res["num_moves"]
    res["frac_mistakes"] = res["num_mistakes"] / res["num_moves"]
    res["frac_inaccuracies"] = res["num_inaccuracies"] / res["num_moves"]
    
    res['username'] = username

    res['last_theory_ply'] = res['last_theory_ply'].fillna(0)

    return res.drop(["moves", "num_blunders", "num_mistakes", "num_inaccuracies"], axis=1)


def avg_move_time(row, username):
    clocks = row["clocks"]
    if not isinstance(clocks, list) or len(clocks) < 2:
        return np.nan

    is_white = row["white.user.name"] == username
    mine = clocks[0::2] if is_white else clocks[1::2]

    initial = row["initial"] * 100
    inc = row["increment"] * 100

    prev = [initial] + mine[:-1]
    times = [p - c + inc for p, c in zip(prev, mine)]
    return np.mean(times) / initial


def calculate_aggregate_game_stats(df):
    latest_game_time = df["createdAt"].max()
    
    latest_game_rating = df[df["createdAt"] == latest_game_time]['rating'].values[0]
        
    averages = df[["acpl", "accuracy", "num_moves", "frac_blunders", "frac_mistakes", "frac_inaccuracies", "avg_move_time", "last_theory_ply"]].mean()
    averages["target_rating"] = latest_game_rating
    
    averages["username"] = df.at[0, "username"]
    
    return averages


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