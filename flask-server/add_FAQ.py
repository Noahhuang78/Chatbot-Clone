import requests
from bs4 import BeautifulSoup
import json
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time
from concurrent.futures import ThreadPoolExecutor
import unicodedata


BASE_URL = "https://www.deltaww.com"
url = f"{BASE_URL}/en-US/FAQ/"


NEW_FAQS = []            #to store newly scraped FAQs
old_faqs = []           #load old FAQs from delta_faq.jsonl

stop_scraping = False     #flags for exiting scraping and checking if need to write to jsonl file
have_newFAQ = False


def load_old():
    global old_faqs
    with open("delta_faq.jsonl", "r", encoding="utf-8") as f: 
            old_faqs = f.readlines()

def normalize(text):
    text = text.strip()
    text = text.lower()
    text = " ".join(text.split())
    text = unicodedata.normalize("NFKC", text)

def new_FAQ(question, answer):                                  #OBSERVATION: We see that new questions are added on the first page of Delta FAQ from the top.
                  #1.load jsonl, 2. if scraped question doesn't match old jsonl data, means new question thus return it
        question = question.strip()

        for line in old_faqs:                                                        
            faq_dict = json.loads(line)    
            print("QUESTION:" + "\n" + repr(question))            #repr() returns raw string representation with white spaces and unicode characters.
            print("OLD QUESTION:"  + "\n" + repr(faq_dict["question"]))
            print("IS OLD QUESTION?" + "\n" +  str(question == faq_dict["question"]))
            print("ANSWER:" + "\n" + repr(answer))
            print("OLD ANSWER:"  + "\n" + repr(faq_dict["answer"]))
            print("IS OLD ANSWER?" + "\n" + str(answer == faq_dict["answer"]))
            if faq_dict["question"].strip() != question.strip() and faq_dict["answer"].strip() != answer.strip(): 
                new_faq = {"question": question, "answer": answer}
                return new_faq
            else:
                return False      #once there is a match in question (meaning we have just reached the section of old questions on FAQ page), we know we have finished going through all the new FAQs based on the OBSERVATION.

def scrap_new():
    options = webdriver.ChromeOptions()
    options.add_argument("--headless") #runs the cSelenium Chrome browser in background without opening Chrome
    driver = webdriver.Chrome(options=options)

    driver.get(url)
    time.sleep(3)

    # Step 1: Load main FAQ page
    # r = requests.get(url)  #get raw html
    soup = BeautifulSoup(driver.page_source, "html.parser") # load html string as soup tree object

    # Step 2: Find all FAQ links
    faq_list = soup.find("ul", class_="faq-list list-wrapper")  #searches the soup tree for the corresponding element and class

    print(faq_list)
    global stop_scraping
    global have_newFAQ
    load_old()
    for i in range(1,100):  #search through up to 99 FAQ pages if Delta ever reaches that amount XD
        if stop_scraping:
            break
        try:
            link = driver.find_element(By.XPATH, f'//ul[@id="pager"]//a[text()="{i}"]')
            link = WebDriverWait(driver, 3).until(             #wait until element is loaded to be clicked, otherwise raises an Exception
        EC.element_to_be_clickable((By.XPATH, f'//ul[@id="pager"]//a[text()="{i}"]'))
    )
            driver.execute_script("arguments[0].click();", link)  # args[0] = link, forces click on link without being in scrollview

    #       ---             for scrollview clicking       ---
    #       driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", link)
    #       link.click()

            time.sleep(2)     #give 2 seconds for page to load
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
                    new_faq = new_FAQ(question, answer)
                    if new_faq == False:               #if no more new faq, --THE-END-- 
                        stop_scraping = True
                        break
                    else:                               #else append to NEW_FAQS and continue searching for new FAQs
                        NEW_FAQS.append(new_faq)
                        print(NEW_FAQS)
                        have_newFAQ = True
                        continue

    
        except Exception as e:
            print(f"Could not click page {i}:{e}")
            break

    if have_newFAQ == True:
        with open("delta_faq.jsonl", "w", encoding="utf-8") as f:
            for faq in NEW_FAQS:                        #we rewrite our delta_faq.jsonl adding the new faqs.
                json.dump(faq, f, ensure_ascii=False)
                f.write("\n")
            for line in old_faqs:
                data = json.loads(line)
                json.dump(data, f, ensure_ascii=False)              
                f.write("\n")

        with open("new_faq.jsonl", "a", encoding = "utf-8") as nf:    #we append new faqs into new_faqs.jsonl just for logging purposes.
            for faq in NEW_FAQS:
                json.dump(faq, nf, ensure_ascii=False)
                nf.write("\n")

