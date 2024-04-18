import requests
import json
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.wait import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import NoSuchElementException
import time
import random

token = r'TY23Wj1cjgTuSAfi8byYldW2sLpnrQiPkLomyfMoSdY%3DWLPb6uBpV%2FNsKyUOjJCK%2BJDeg0YxYEhyhE2KyGcQwcoTedbJcTkOyt0L6GvRHaTIYlj3kfOfunYpe%2BUT3MJtYRi%2BclxqtfQFW4tSsIiJFo0%3D'

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
                "lng": lng
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
            retval = session.post(f'https://game-server.geoguessr.com/api/{game_mode}/{game_id}/guess', json=guess_data)

        # print(retval.text)
            
        print(f'Guess submitted for round {round_number}')
        
        if game_mode == 'country_streak':
            driver.refresh()
            wait = WebDriverWait(driver, 20)
            wait.until(EC.element_to_be_clickable((By.XPATH, "/html/body/div[1]/div[2]/div[2]/main/div[2]/div[1]/div[2]/div/div[2]/div/div/div[4]/div/div/button"))).click()
            game_id = driver.current_url.split('/')[-1]
            current_round = 0

        
        if game_mode == 'world':
            driver.refresh()
            game_id = driver.current_url.split('/')[-1]
            current_round = 0

        else:
            driver.refresh()
            current_round = round_number + 1

        return current_round

    def join_game(self, game_id, session):
        driver = webdriver.Firefox()
        driver.get('https://www.geoguessr.com/')
        driver.add_cookie({'name':'_ncfa', 'value':token})
        driver.refresh()
        url = f"https://www.geoguessr.com/join/{game_id}?s=Url"
        driver.get(url)
        session, isProUser = self.get_session(token)
        driver.refresh()
        wait = WebDriverWait(driver, 20)
        driver.maximize_window()
        wait.until(EC.element_to_be_clickable((By.XPATH, '//*[@id="accept-choices"]'))).click()

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
            time.sleep(0.5)
        game_id = driver.current_url.split('/')[-1]
        
        return driver, game_id, session, isProUser, game_mode

    def remove_html_element(self, driver, xpath):
        try:
            driver.execute_script("arguments[0].remove();", driver.find_element(By.XPATH, xpath))
            print(f"Removed {xpath}")
        except NoSuchElementException:
            pass


    def take_screenshots(self, driver, filename, game_id):

        # removing elements
        driver.execute_script("arguments[0].remove();", driver.find_element(By.XPATH, "/html/body/div[1]/div[2]/div[2]/main/div/div/aside/div")) # left side controls
        driver.execute_script("arguments[0].remove();", driver.find_element(By.XPATH, "/html/body/div[1]/div[2]/div[2]/main/div/div/div[2]")) # top hud
        driver.execute_script("arguments[0].remove();", driver.find_element(By.XPATH, "/html/body/div[1]/div[2]/div[2]/main/div/div/div[2]/div/div[2]")) # top right (streak counter)
        driver.execute_script("arguments[0].remove();", driver.find_element(By.XPATH, "/html/body/div[1]/div[2]/div[2]/main/div/div/div[3]/div")) # map
        driver.execute_script("arguments[0].remove();", driver.find_element(By.XPATH, "/html/body/div[1]/div[2]/div[2]/main/div/div/div[1]/div/div/div/div/div[2]/div[1]/div[10]")) # arrows
        driver.execute_script("arguments[0].remove();", driver.find_element(By.XPATH, "/html/body/div[1]/div[2]/div[2]/main/div/div/div[1]/div/div/div/div/div[16]/div")) # arrows
        
        action = webdriver.ActionChains(driver)
        element = driver.find_element(By.XPATH, '/html/body/div[1]/div[2]/div[2]/main/div/div/div[1]/div/div/div/div/div[2]/div[1]/div[9]/div/div/canvas') # or your another selector here
        filenames = []
        for i in range(4):
            action.drag_and_drop_by_offset(element, 750, 0).perform()   # 700px to the right, 0px to bottom
            time.sleep(0.2)
            filepath = f'imgs/{game_id}/{filename}_{i+1}.png'
            driver.save_screenshot(filepath)
            filenames.append(filepath)
        
        return filenames


    def play_round(self, driver, game_mode, game_id, session, new_round):
        print("playing round")
        try:
            wait = WebDriverWait(driver, 20)
            wait.until(EC.presence_of_element_located((By.XPATH, '/html/body/div/div[2]/div[2]/main/div/div/div[4]/div/div[3]/div/div/div/div[3]/div[1]/div[2]'))) ### dont think it works
        except NoSuchElementException:
            print("Map not loaded")
            time.sleep(1)

        current_round = new_round
        print(f'Current round: {current_round}')

        filename = f'round_{current_round}'
        filenames = self.take_screenshots(driver, filename, game_id)

        return filenames, current_round
