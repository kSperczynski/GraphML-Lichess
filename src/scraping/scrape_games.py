from argparse import ArgumentParser
import requests
import json
import os

LICHESS_API_TOKEN = "lip_xzA4lrs8io8L84opqqvX"
NUM_GAMES_DEFAULT = 50
SCRAPING_CHECKPOINTG_RATE = 10
GAMES_FOLDER_DEFAULT = "games"
CHECKPOINT_FOLDER_DEFAULT = "games_checkpoint"
URL = "https://lichess.org/api/games/user/"
OPPONENTS_FOLDER_DEFAULT = "matchups"
PARAMS = {
    "max": NUM_GAMES_DEFAULT,
    "rated": "true",
    "perfType": "rapid",
    "analysed": "true",
    "moves": "true",
    "clocks": "true",
    "evals": "false",
    "accuracy": "true",
    "opening": "true"
}
HEADERS = {
    "Accept": "application/x-ndjson",
    "Authorization": f"Bearer {LICHESS_API_TOKEN}"
}


def main(args):
    queue = None

    if args.uname is not None:
        queue = [args.uname]
    elif args.queue_checkpoint is not None:
        with open(f"{args.queue_checkpoint}/game_scraping_checkpoint.json", 'r') as file: 
            # print(file)
            # data = json.loads(file)
            try:
                data = json.load(file)
                queue = data['queue']
            except:
                print("No username provided and no checkpoint exists")
                raise ValueError("No username provided and no checkpoint exists")
    else:
        print("No username provided and no checkpoint exists")
        raise ValueError("No username provided and no checkpoint exists")
        
    all_added = set()

    already_fetched = get_already_fetched(args.games_folder)

    while len(all_added) < args.num_players:
        if len(queue) > 0:
            username = queue.pop()

            if username in already_fetched or username in all_added:
                continue

            try:
                games = get_games(username, args.num_games)
            except:
                # this usually happens if someone deletes their account
                print(f"Failed to fetch games for {username}")
                continue

            all_added.add(username)

            print(f"Fetched games for {username}. Total {len(all_added)}/{args.num_players} users fetched.")

            write_games(username, games, args.games_folder)

            if len(all_added) % SCRAPING_CHECKPOINTG_RATE == 0 and len(all_added) > 0:
                checkpoint_added_and_queue(args.checkpoint_folder, all_added, queue)

            opponents = []

            for game in games:
                # this happens when a player remained anonymous
                # we cannot add their username to the queue for the BFS
                if (not 'user' in game['players']['white'].keys()) or (not 'user' in game['players']['black'].keys()):
                    continue 

                white = game['players']['white']['user']['name']
                black = game['players']['black']['user']['name']

                if white not in all_added:
                    queue.append(white)
                if (black not in all_added):
                    queue.append(black)

                if white == username:
                    opponents.append(black)
                else:
                    opponents.append(white)

                write_opponents(username, opponents, args.opponents_folder)

        else:
            print("Queue is empty - stopped scraping games")
            checkpoint_added_and_queue(args.checkpoint_folder, all_added, queue)
            return all_added, queue

    checkpoint_added_and_queue(args.checkpoint_folder, all_added, queue)
    return all_added, queue


def get_games(username, limit):
    print(f"Fetching games for {username}")
    PARAMS['max'] = limit

    response = requests.get(f"{URL}{username}", params=PARAMS, headers=HEADERS)
    # raises error if unsuccessful request
    response.raise_for_status()

    games = [json.loads(x) for x in response.text.split("\n")[:-1]]

    return games


def write_games(username, games, games_folder):
    with open(f"{games_folder}/{username}.json", 'w') as file: 
        file.write(json.dumps(games)) 
        file.close() 


def get_already_fetched(games_folder):
    filenames = os.listdir(games_folder)

    return set([file.split('.')[0] for file in filenames])


def checkpoint_added_and_queue(checkpoint_folder, added, queue):
    with open(f"{checkpoint_folder}/game_scraping_checkpoint.json", 'w+') as file: 
        data = {
            "added": list(added),
            "queue": queue
        }

        file.write(json.dumps(data)) 

        file.close() 


def write_opponents(username, opponents, opponents_folder):
    with open(f"{opponents_folder}/{username}.json", 'w') as file: 
        file.write(json.dumps(opponents)) 

        file.close() 


if __name__ == "__main__":
    parser = ArgumentParser(description="Scrape games args")

    parser.add_argument("--uname")
    parser.add_argument(
        "--num_games", 
        type=int, 
        default=NUM_GAMES_DEFAULT
    )
    parser.add_argument(
        "--num_players", 
        type=int, 
        default=100
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

    args = parser.parse_args()

    main(args)
    # TODO: serialise queue and maybe list of guys????
    # TODO: allow checkpointing on queue through filename input