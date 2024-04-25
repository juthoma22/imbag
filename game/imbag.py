import json
import time
import os

from GeoGame import GeoGame
from ImbagClassifier import ImbagClassifier
from GeoClassifier import GeographicalClassifier


if __name__ == '__main__':

    mean_embedding = True

    second_model = False
    
    print("Loading Model...")
    classifier = ImbagClassifier()

    while True:

        print("Starting Game...")
        game = GeoGame()

        game_mode = 'duels_party'

        if game_mode == 'duels_party':
            party_code = "6XDP"
            driver, game_id, session, isProUser = game.join_game(party_code)

        else:
            driver, game_id, session, isProUser, game_mode = game.start_game(game_mode)

        if not second_model:
            os.mkdir(f"imgs/{game_id}")

        if game_mode == 'duels':
            url = f'https://game-server.geoguessr.com/api/duels/{game_id}'
            status_keyword = 'status'
            round_keyword = 'currentRoundNumber'
            finished_keyword = 'Game finished'

        if game_mode == 'country_streak':
            url = f'https://geoguessr.com/api/v3/games/{game_id}'
            status_keyword = 'state'
            round_keyword = 'roundCount'

        if game_mode == 'world':
            url = f'https://geoguessr.com/api/v3/games/{game_id}'
            status_keyword = 'state'
            round_keyword = 'round'
            finished_keyword = 'finished'

        if game_mode == 'duels_party':
            url = f'https://game-server.geoguessr.com/api/duels/{game_id}'
            status_keyword = 'status'
            round_keyword = 'currentRoundNumber'
            finished_keyword = 'Finished'

        current_round = None

        while True:

            print(url)
            print(f"Game ID: {game_id}")
            response = session.get(url)
            data = json.loads(response.text)
            # print(data)

            game_status = data.get(status_keyword)
            print(game_status)

            print(f"Game status: {game_status}")

            if game_mode != 'country_streak' and finished_keyword in game_status:
                print("Game finished")
                # write final score to file
                if not second_model:
                    if game_mode == 'duels_party':
                        score_file = "final_score_aow_party.txt"
                    else:
                        if mean_embedding:
                            score_file = "final_score_aow_mean_embedding_360.txt"
                        else:
                            score_file = "final_score_aow_360.txt"

                    with open(score_file, "a+") as f:
                        f.write(f"{game_id}\n")
                break

            # print(data)

            new_round = data.get(round_keyword)

            print(f"New round: {new_round}")

            if new_round != current_round:
                current_round = new_round
                print(new_round, current_round)
                print(game_mode)
                filenames = game.play_round(driver, game_mode, game_id, session, new_round, isProUser, second_model)

                country, lat, lon, geocell, values = classifier.make_prediction(filenames, mean_embedding=mean_embedding)
                print(f"Country: {country}, Latitude: {lat}, Longitude: {lon}")
                print(f"Guess: {lat}, {lon} in Geocell {geocell}")

                game.guess_location(game_mode, session, game_id, lat, lon, new_round, driver)
            else:
                print("Waiting for next round...")
                time.sleep(3)

        driver.quit()

