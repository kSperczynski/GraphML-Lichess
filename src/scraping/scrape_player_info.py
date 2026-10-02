from argparse import ArgumentParser
import requests
import json
import os

LICHESS_API_TOKEN = "lip_xzA4lrs8io8L84opqqvX"
GAMES_FOLDER_DEFAULT = "games"
PLAYERS_FOLDER_DEFAULT = "players"
HEADERS = {
    "Accept": "application/x-ndjson",
    "Authorization": f"Bearer {LICHESS_API_TOKEN}"
}


def main(args):
    already_fetched = set([filename.split('.')[0] for filename in os.listdir(args.players_folder)])

    for filename in os.listdir(args.games_folder):
        username = filename.split('.')[0]

        if username in already_fetched:
            continue

        try:
            player = get_user_info(username)

            write_player(username, player, args.players_folder)
        except:
            print(f"User {username} not found")


def get_user_info(username):
    URL = f"https://lichess.org/api/user/{username}/perf/rapid"

    print(f"Fetching {username}")

    response = requests.get(URL, params={}, headers=HEADERS)
    response.raise_for_status()

    return response.json()


def write_player(username, player, players_folder):
    with open(f"{players_folder}/{username}.json", 'w') as file: 
        file.write(json.dumps(player)) 
        file.close() 


if __name__ == "__main__":
    parser = ArgumentParser(description="Scrape games args")

    parser.add_argument(
        "--players_folder",
        default=PLAYERS_FOLDER_DEFAULT
    )
    parser.add_argument(
        "--games_folder",
        default=GAMES_FOLDER_DEFAULT
    )

    args = parser.parse_args()

    main(args)
    