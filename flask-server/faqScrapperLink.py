import requests
from bs4 import BeautifulSoup
import json
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait, Select
from selenium.webdriver.support import expected_conditions as EC
import time
from concurrent.futures import ThreadPoolExecutor


BASE_URL = "https://www.deltaww.com"
url = f"{BASE_URL}/en-US/service-support/faq"

driver = webdriver.Chrome()
driver.get(url)
time.sleep(3)

seen_questions = set()
faqs_data = []

for options_index in range(100):
    try:
        dropdown_1 = driver.find_element(By.CLASS_NAME, "MuiSelect-select MuiSelect-outlined MuiInputBase-input MuiOutlinedInput-input css-1xk9oh3")
        select_1 = Select(dropdown_1)
        select_1.select_by_visible_text("Industrial Automation")
        dropdown_2 = driver.find_element(By.ID, "DropDownList2")
        select_2 = Select(dropdown_2)
        select_2.select_by_index(options_index)
        submit = WebDriverWait(driver, 3).until(EC.element_to_be_clickable((By.ID, "send_btn")))
        driver.execute_script("arguments[0].click()", submit)

    except Exception as e:
        print(f"no more selectable options: {e}")
        break

    time.sleep(2)

    for page in range(2,20):
        try:             #wait object that keeps checking until 3 seconds. .until(...) waits until condition inside is true or if time runs out.
           
            url = driver.current_url
            print(url)
            soup = BeautifulSoup(driver.page_source, "html.parser")
            faq_list = soup.find("ul", class_="faq-list list-wrapper")
            for li in faq_list.find_all("li"):
                a = li.find("a", href=True)
                question = a.get_text(strip=True) if a else ""
                print(f"Question: {question}")
                if a:
                    link = BASE_URL + a['href']
                    raw = requests.get(link)
                    new_soup = BeautifulSoup(raw.text, "html.parser")
                    answer_div = new_soup.find("div", class_="this-info") 
                    answer = answer_div.get_text(separator=" ", strip=True) if answer_div else "No Answer"   #join child nodes with a space e.g Hello<b>World</b> would be "Hello World" not "HelloWorld".
                    if question not in seen_questions:
                        print(question)
                        faqs_data.append({"question": question, "answer": answer, "url": url})
                        seen_questions.add(question)
                    print(faqs_data)

            page_select = WebDriverWait(driver, 3).until(EC.element_to_be_clickable((By.XPATH, f'//ul[@id="pager"]//a[text()="{page}"]')))  #if element is clickable in 3s, store in page_select
            driver.execute_script("arguments[0].click()", page_select)
            print(f"Scaping Page: {page}")
        except Exception as e:
            print(f"{e}")
            break
        
    
driver.quit()   

with open("delta_faq.jsonl", "w", encoding="utf-8") as f:
    for faq_data in faqs_data:
        json.dump(faq_data, f, ensure_ascii=False)
        f.write("\n")
    