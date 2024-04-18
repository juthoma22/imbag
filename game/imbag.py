import json
import time
import os

from GeoGame import GeoGame
from ImbagClassifier import ImbagClassifier
from GeoClassifier import GeographicalClassifier


if __name__ == '__main__':

    mean_embedding = False
    
    print("Loading Model...")
    classifier = ImbagClassifier()

    while True:

        print("Starting Game...")
        game = GeoGame()

        game_mode = 'duels'

        while True:
            
            if game_mode == 'duels_party':
                game_id = "abc"
                driver, game_id, session, isProUser = game.join_game(game_id, session)

            else:
                driver, game_id, session, isProUser, game_mode = game.start_game(game_mode)

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
                finished_keyword = 'Game finished'

            os.mkdir(f"imgs/{game_id}")

            current_round = None


            response = session.get(url)
            data = json.loads(response.text)

            game_status = data.get(status_keyword)

            print(f"Game status: {game_status}")

            if game_mode != 'country_streak' and finished_keyword in game_status:
                final_score = data["player"]["totalScore"]["amount"]
                total_distance = data["player"]["totalDistance"]["meters"]["amount"]
                print("Game finished")
                print(f"Final score: {final_score}")
                # write final score to file
                if mean_embedding:
                    score_file = "final_score_aow_mean_embedding.txt"
                else:
                    score_file = "final_score_aow.txt"

                with open(score_file, "a+") as f:
                    f.write(f"{game_id}, {final_score}, {total_distance}\n")
                break

            # print(data)

            new_round = data.get(round_keyword)

            print(f"New round: {new_round}")

            if new_round != current_round:
                filenames, current_round = game.play_round(driver, game_mode, game_id, session, new_round)

                country, lat, lon, geocell, values = classifier.make_prediction(filenames, mean_embedding=mean_embedding)
                print(f"Country: {country}, Latitude: {lat}, Longitude: {lon}")
                print(f"Guess: {lat}, {lon} in Geocell {geocell}")

                current_round = game.guess_location(game_mode, session, game_id, lat, lon, current_round, driver) ### only works for duels

        driver.quit()

