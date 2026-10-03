from argparse import ArgumentParser
import requests
import json
import os
import time

import pandas as pd

LICHESS_API_TOKEN = "lip_xzA4lrs8io8L84opqqvX"
NUM_GAMES_DEFAULT = 10
SCRAPING_CHECKPOINTG_RATE = 10
GAMES_FOLDER_DEFAULT = "games"
CHECKPOINT_FOLDER_DEFAULT = "games_checkpoint"
URL = "https://lichess.org/api/games/user/"
OPPONENTS_FOLDER_DEFAULT = "matchups"
MATCHUP_RESULTS_FOLDER_DEFAULT = "head-to-head-results"
PARAMS = {
    "max": NUM_GAMES_DEFAULT,
    "rated": "true",
    "perfType": "rapid",
    "moves": "false",
    "clocks": "false",
    "evals": "false",
    "accuracy": "false",
    "opening": "false"
}
HEADERS = {
    "Accept": "application/x-ndjson",
    "Authorization": f"Bearer {LICHESS_API_TOKEN}"
}


def main(args):
    already_fetched = set([filename.split('.')[0] for filename in os.listdir(args.matchup_results_folder)])
    matchups = set([filename.split('.')[0] for filename in os.listdir(args.opponents_folder)])

    for filename in os.listdir(args.games_folder):
        username = filename.split('.')[0]

        if username in already_fetched or username not in matchups:
            continue

        print(f"Generating edges for {username}")

        opponents = []

        try:
            with open(f"{args.opponents_folder}/{username}.json", 'r') as file: 
                opponents = json.load(file)
                file.close()
        except:
            print(f"Matchups file for {username} not found locally")

        if len(opponents) == 0:
            continue

        froms = []
        tos = []
        from_points = []

        for opponent in opponents:
            try:
                all_games_with_opponent = get_all_games_with_opponent(username, opponent)
            except Exception as e:
                print(f"Could not fetch games {username} played againt {opponent}")
                print(e)
                continue

            this_points = 0
            opponent_points = 0

            for game in all_games_with_opponent:
                if "winner" not in game:
                    this_points += 0.5
                    opponent_points += 0.5
                else:
                    winner = game['winner']

                    if game['players'][winner]['user']['name'] == username:
                        this_points += 1
                    elif game['players'][winner]['user']['name'] == opponent:
                        opponent_points += 1
                    else:
                        print("Invalid winner for this game")
                        print(game)
                        print(username)
                        print(opponent)
                    # print(winner)
                    # print(game)
                    # return
            if this_points > 0:
                froms.append(username)
                tos.append(opponent)
                from_points.append(this_points)

            if opponent_points > 0:
                froms.append(opponent)
                tos.append(username)
                from_points.append(opponent_points)

        df = pd.DataFrame({
            "from": froms,
            "to": tos,
            "points": from_points
        })

        df.to_csv(f"{args.matchup_results_folder}/{username}.csv", index=False)
        print("--------------------------------------------------------------------")


def get_all_games_with_opponent(username, opponent):
    print(f"Fetching games for {username} VS {opponent}")
    attempts = 0
    games = None

    while attempts < 3:
        PARAMS['vs'] = opponent

        response = requests.get(f"{URL}{username}", params=PARAMS, headers=HEADERS)
        if response.status_code == 429:
            attempts += 1
            print("Rate limit exceeded, trying again for 10 seconds")
            time.sleep(10)
        # raises error if unsuccessful request
        else:
            response.raise_for_status()

            games = [json.loads(x) for x in response.text.split("\n")[:-1]]

            break

    if games is None:
        raise Exception("No games fetched")

    return games


if __name__ == "__main__":
    parser = ArgumentParser(description="Scrape games args")

    parser.add_argument("--uname")
    parser.add_argument(
        "--num_games", 
        type=int, 
        default=NUM_GAMES_DEFAULT
    )
    parser.add_argument(
        "--games_folder", 
        default=GAMES_FOLDER_DEFAULT
    )
    parser.add_argument(
        "--checkpoint_folder", 
        default=CHECKPOINT_FOLDER_DEFAULT
    )
    parser.add_argument(
        "--queue_checkpoint"
    )
    parser.add_argument(
        "--opponents_folder",
        default=OPPONENTS_FOLDER_DEFAULT
    )
    parser.add_argument(
        "--matchup_results_folder",
        default=MATCHUP_RESULTS_FOLDER_DEFAULT
    )

    args = parser.parse_args()

    main(args)
    # TODO: serialise queue and maybe list of guys????
    # TODO: allow checkpointing on queue through filename input