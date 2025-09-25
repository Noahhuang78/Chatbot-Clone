import requests
from bs4 import BeautifulSoup
import json
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time
from concurrent.futures import ThreadPoolExecutor


BASE_URL = "https://www.deltaww.com"
url = f"{BASE_URL}/en-US/FAQ/"

options = webdriver.ChromeOptions()
# options.add_argument("--headless") #runs the cSelenium Chrome browser in background without opening Chrome
driver = webdriver.Chrome(options=options)

driver.get(url)
time.sleep(3)

# Step 1: Load main FAQ page
# r = requests.get(url)  #get raw html
soup = BeautifulSoup(driver.page_source, "html.parser") # load html string as soup tree object

# Step 2: Find all FAQ links
faq_list = soup.find("ul", class_="faq-list list-wrapper")  #searches the soup tree for the corresponding element and class

FAQ_data = []
print(faq_list)



for i in range(1,100):  #58 FAQ pages
    try:
        link = driver.find_element(By.XPATH, f'//ul[@id="pager"]//a[text()="{i}"]')
        link = WebDriverWait(driver, 3).until(             #wait until element is loaded to be clicked
    EC.element_to_be_clickable((By.XPATH, f'//ul[@id="pager"]//a[text()="{i}"]'))
)
        driver.execute_script("arguments[0].click();", link)  # forces click on link without being in scrollview

#       ---             for scrollview clicking       ---
#       driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", link)
#       link.click()

        time.sleep(2)
        print(f"Page {i} loaded")

        soup = BeautifulSoup(driver.page_source, "html.parser")
        faq_list = soup.find("ul",class_="faq-list list-wrapper" )

        for li in faq_list.find_all("li"):       #filters all "li" elements in faq list, run a for loop through the filtered list.
        
            a = li.find("a", href=True)          #find anchor tag elements with href in each li.
            question = a.get_text(strip=True) if a else "No Question"
            if a:                                #if found, get the question and answer text and append to FAQ_data as "question" and "answer" key-value pairs dictionary. 
                link = BASE_URL + a['href']
                raw = requests.get(link)
                faq_soup = BeautifulSoup(raw.text, 'html.parser')
                answer_div = faq_soup.find("div", class_= "this-info")
                answer = answer_div.get_text(" ",strip=True) if answer_div else "No Answer"
                FAQ_data.append({'question': question, 'answer': answer})
 
        

    except Exception as e:
        print(f"Could not click page {i}:{e}")
        break
    
driver.quit()

with open("delta_faq.jsonl", "w", encoding= "utf-8") as f:
    for data in FAQ_data:
        json.dump(data, f, ensure_ascii=False)
        f.write("\n")


