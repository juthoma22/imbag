import requests
import json
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.wait import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import NoSuchElementException, TimeoutException
import time
import random

# token = r'TY23Wj1cjgTuSAfi8byYldW2sLpnrQiPkLomyfMoSdY%3DWLPb6uBpV%2FNsKyUOjJCK%2BJDeg0YxYEhyhE2KyGcQwcoTedbJcTkOyt0L6GvRHaTIYlj3kfOfunYpe%2BUT3MJtYRi%2BclxqtfQFW4tSsIiJFo0%3D' # juthoma
token = r'lQ3WYhyUoftGEuqJRUIP3uTF%2BlYewaXMc3bvkjBA1eo%3DBzWlf4bDQsA6mwj0I7bklqRlLhG%2F77KQxiKbjbLpLnhtaO1tVorpRricStcxJuJKCa4up5CycmSptBlb%2BMK0Cy7m4NWcJcp2U0GRn%2F2Jc%2F4%3D' # imbag

class GeoGame:
    
    def wait(self, driver, timeout=20):
        WebDriverWait(driver, timeout)

    def get_session(self, ncfa_token):
        BASE_URL = "https://www.geoguessr.com/api/" 
        _ncfa_TOKEN = ncfa_token
        session = requests.Session()
        session.cookies.set("_ncfa", _ncfa_TOKEN, domain="www.geoguessr.com")

        profiles = session.get(f"{BASE_URL}v3/profiles")
        isProUser = json.loads(profiles.text).get('user').get('isProUser')
        if profiles.status_code != 200:
            Exception(f"Error: {profiles.status_code}")
        return session, isProUser

    def get_coordinates_for_guess(self):
        upper_left = (53.915173, -2.173165)
        lower_left = (37.124868, -8.468529)
        lower_right = (42.340269, 26.894492)
        upper_right = (63.751013, 28.385732)
        
        random_latitude = random.uniform(lower_left[0], upper_left[0])
        random_longitude = random.uniform(lower_right[1], upper_right[1])

        # vienna
        random_latitude = 48.2082
        random_longitude = 16.3738
        return (random_latitude, random_longitude)

    def guess_location(self, game_mode, session, game_id, lat, lng, round_number, driver):
        guess_data = {
                "lat": lat,
                "lng": lng,
                "roundNumber": round_number,
                }
        
        if game_mode == 'duels':
            guess_data['roundNumber'] = round_number
        
        if game_mode == 'country_streak':
            guess_data = {
                "streakLocationCode" : "fr"
            }
            retval = session.post(f'https://www.geoguessr.com/api/v3/games/{game_id}', json=guess_data)

        if game_mode == 'world':
            retval = session.post(f'https://www.geoguessr.com/api/v3/games/{game_id}', json=guess_data)

        else:
            retval = session.post(f'https://game-server.geoguessr.com/api/duels/{game_id}/guess', json=guess_data)

        print(f'Guess submitted for round {round_number}')
        
        if game_mode == 'country_streak':
            driver.refresh()
            wait = WebDriverWait(driver, 20)
            wait.until(EC.element_to_be_clickable((By.XPATH, "/html/body/div[1]/div[2]/div[2]/main/div[2]/div[1]/div[2]/div/div[2]/div/div/div[4]/div/div/button"))).click()

        else:
            driver.refresh()

    def join_game(self, party_code):
        driver = webdriver.Firefox()
        driver.get('https://www.geoguessr.com/')
        driver.add_cookie({'name':'_ncfa', 'value':token})
        driver.refresh()
        url = f"https://www.geoguessr.com/join/{party_code}?s=Url"
        driver.get(url)
        session, isProUser = self.get_session(token)
        driver.refresh()
        wait = WebDriverWait(driver, 20)
        driver.maximize_window()
        wait.until(EC.element_to_be_clickable((By.XPATH, '//*[@id="accept-choices"]'))).click()

        url = "https://www.geoguessr.com/party"

        while url in driver.current_url:
            time.sleep(0.5)
        game_id = driver.current_url.split('/')[-1]

        return driver, game_id, session, isProUser


    def start_game(self, game_mode):

        if game_mode == 'country_streak':
            
            xpath = "/html/body/div[1]/div[2]/div[2]/div[1]/main/div/div/div/div/article/div[2]/div[1]/div/button" # play game button
            xpath_non_pro = "/html/body/div[1]/div[4]/div/div/div[4]/div/button" # non pro user accept 15 minutes
            url = 'https://www.geoguessr.com/country-streak'

        if game_mode == 'duels':

            xpath = '/html/body/div[1]/div[4]/div/div/div[4]/div/button'
            xpath_non_pro = "'/html/body/div[1]/div[2]/div[2]/div[1]/main/div[2]/div[3]/div[1]/div/div[3]/div[2]/div[3]/div/button/div/div/div[1]'"
            url = 'https://www.geoguessr.com/multiplayer'
        
        if game_mode == 'world':
            
            xpath = "/html/body/div/div[2]/div[2]/div[1]/main/div/div/div/div/div[3]/div/div/button" # play game button
            xpath_non_pro = "/html/body/div[1]/div[4]/div/div/div[4]/div/button" # non pro user accept 15 minutes
            url = 'https://www.geoguessr.com/maps/652ba0d9002aa0d36f996153/play'

        driver = webdriver.Firefox()
        driver.get('https://www.geoguessr.com/')
        driver.add_cookie({'name':'_ncfa', 'value':token})
        driver.refresh()
        driver.get(url)
        session, isProUser = self.get_session(token)
        driver.refresh()

        wait = WebDriverWait(driver, 20)
        driver.maximize_window()
        wait.until(EC.element_to_be_clickable((By.XPATH, '//*[@id="accept-choices"]'))).click()

        time.sleep(1) # wait for page to load - cannot change i think
        wait.until(EC.element_to_be_clickable((By.XPATH, xpath))).click()
        if not isProUser:
            time.sleep(1)
            wait.until(EC.element_to_be_clickable((By.XPATH, xpath_non_pro))).click()
        
        while url in driver.current_url:
            time.sleep(0.1)
        game_id = driver.current_url.split('/')[-1]
        
        return driver, game_id, session, isProUser, game_mode

    def remove_html_element(self, driver, xpath):
        try:
            driver.execute_script("arguments[0].remove();", driver.find_element(By.XPATH, xpath))
            print(f"Removed {xpath}")
        except NoSuchElementException:
            pass


    def take_screenshots(self, driver, filename, game_id, game_mode, is_pro, second_model):

        # removing elements
        print("Removing elements")
        print(game_mode, is_pro)
        time.sleep(0.2)
        if is_pro and game_mode == 'duels_party':
            driver.execute_script("arguments[0].remove();", driver.find_element(By.XPATH, "/html/body/div/div[2]/div[2]/main/div/div/aside[2]/div")) # left side controls
            driver.execute_script("arguments[0].remove();", driver.find_element(By.XPATH, "/html/body/div/div[2]/div[2]/main/div/div/div[1]/div")) # top hud
            driver.execute_script("arguments[0].remove();", driver.find_element(By.XPATH, "/html/body/div/div[2]/div[2]/main/div/div/aside[1]/div")) # chat & emotes
            if game_mode == 'duels_party':
                driver.execute_script("arguments[0].remove();", driver.find_element(By.XPATH, "/html/body/div[1]/div[2]/div[2]/main/div/div/div[3]/div")) # map
                element = driver.find_element(By.XPATH, '/html/body/div/div[2]/div[2]/main/div/div/div[2]/div/div/div/div/div[2]/div[1]/div[9]/div/div/canvas[1]') # or another selector here
                driver.execute_script("arguments[0].remove();", driver.find_element(By.XPATH, "/html/body/div/div[2]/div[2]/main/div/div/div[2]/div/div/div/div/div[2]/div[1]/div[10]")) # arrows
        elif is_pro and game_mode == 'world':
            driver.execute_script("arguments[0].remove();", driver.find_element(By.XPATH, "/html/body/div[1]/div[2]/div[2]/main/div/div/aside/div")) # left side controls
            driver.execute_script("arguments[0].remove();", driver.find_element(By.XPATH, "/html/body/div/div[2]/div[2]/main/div/div/div[2]")) # top hud
            driver.execute_script("arguments[0].remove();", driver.find_element(By.XPATH, "/html/body/div[1]/div[2]/div[2]/main/div/div/div[2]/div")) # top right
            driver.execute_script("arguments[0].remove();", driver.find_element(By.XPATH, "/html/body/div[1]/div[2]/div[2]/main/div/div/div[3]")) # map
            driver.execute_script("arguments[0].remove();", driver.find_element(By.XPATH, "/html/body/div[1]/div[2]/div[2]/main/div/div/div[1]/div/div/div/div/div[2]/div[1]/div[10]")) # arrows
            element = driver.find_element(By.XPATH, '/html/body/div[1]/div[2]/div[2]/main/div/div/div[1]/div/div/div/div/div[2]/div[1]/div[9]/div/div/canvas[1]')

        print("Removed elements")
        print("Taking screenshots")
        action = webdriver.ActionChains(driver)
        filenames = []
        filepath = f'imgs/{game_id}/{filename}_{1}.png'
        driver.save_screenshot(filepath)
        filenames.append(filepath)
        for i in range(2):  # 700px to the right, 0px to bottom
            x = 400
            action.drag_and_drop_by_offset(element, x, 0).perform()
            action.drag_and_drop_by_offset(element, x, 0).perform()
            action.drag_and_drop_by_offset(element, x, 0).perform()
            time.sleep(0.2) 
            filepath = f'imgs/{game_id}/{filename}_{i+2}.png'
            if not second_model:
                driver.save_screenshot(filepath)
            filenames.append(filepath)

        driver.refresh()
        
        return filenames


    def play_round(self, driver, game_mode, game_id, session, round, is_pro, second_model=False):
        print("playing round")
        if game_mode == 'duels_party':
            if is_pro:
                x_path = "/html/body/div/div[2]/div[2]/main/div/div/div[3]/div/div[3]/div/div/div/div[3]/div[1]/div[2]"
            else:
                x_path = "/html/body/div[1]/div[2]/div[3]/main/div/div/div[3]/div/div[3]/div/div/div/div[3]/div[1]/div[2]"

        else:
            x_path = "/html/body/div/div[2]/div[2]/main/div/div/div[4]/div/div[3]/div/div/div/div[3]/div[1]/div[2]"
        
        print("Waiting for round to load...")
        try:
            wait = WebDriverWait(driver, 50)
            wait.until(EC.presence_of_element_located((By.XPATH, x_path))) ### dont think it works
        except TimeoutException:
            TimeoutException("Map not loaded")

        print("Round loaded")

        filename = f'round_{round}'
        filenames = self.take_screenshots(driver, filename, game_id, game_mode, is_pro, second_model)

        return filenames
