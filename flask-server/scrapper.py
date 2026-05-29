from selenium import webdriver
from selenium.webdriver.common.by import By
import time
import pickle

#load cookies for signing into jobstreet account
driver = webdriver.Chrome()
driver.get("https://sg.jobstreet.com/")
time.sleep(30)
cookies = driver.get_cookies()
print(cookies)
driver.close()
with open("cookies.pkl", "wb") as f:
    pickle.dump(cookies, f)





