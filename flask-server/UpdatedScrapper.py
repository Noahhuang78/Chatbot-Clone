import json
import time
import requests
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

BASE_URL = "https://www.deltaww.com"
URL = f"{BASE_URL}/en-US/service-support/faq"

driver = webdriver.Chrome()
driver.get(URL)
wait = WebDriverWait(driver, 10)

seen_questions = set()
faqs_data = []

def get_mui_options(dropdown_css_selector):
    """Clicks an MUI dropdown and returns the list of option elements."""
    dropdown = wait.until(EC.element_to_be_clickable((By.CSS_SELECTOR, dropdown_css_selector)))
    dropdown.click()
    # MUI menus are usually <li> elements with role="option"
    return wait.until(EC.presence_of_all_elements_located((By.CSS_SELECTOR, "li[role='option']")))

try:
    # 1. Handle the first dropdown (Product Category)
    # We'll just select "Industrial Automation" once
    cat_options = get_mui_options(".MuiSelect-select") # The first MUI select
    for opt in cat_options:
        if "Industrial Automation" in opt.text:
            opt.click()
            break
    
    time.sleep(1) # Brief pause for second dropdown to populate

    # 2. Loop through the second dropdown (Sub-category)
    # Since the DOM refreshes, we might need to re-open this inside the loop
    for i in range(100):
        try:
            # Re-open the second dropdown (it's the second element with that class)
            sub_dropdowns = driver.find_elements(By.CSS_SELECTOR, ".MuiSelect-select")
            if len(sub_dropdowns) < 2: break 
            sub_dropdowns[1].click()
            
            # Get fresh list of options
            options = wait.until(EC.presence_of_all_elements_located((By.CSS_SELECTOR, "li[role='option']")))
            
            if i >= len(options):
                break
                
            current_option = options[i]
            option_text = current_option.text
            current_option.click()
            
            # Click the Search/Submit button
            submit = wait.until(EC.element_to_be_clickable((By.ID, "send_btn")))
            driver.execute_script("arguments[0].click();", submit)
            
            # Wait for results to load
            time.sleep(2)

            # 3. Pagination & Scraping
            for page in range(1, 20): # Start at 1 for logic
                soup = BeautifulSoup(driver.page_source, "html.parser")
                faq_list = soup.find("ul", class_="faq-list list-wrapper")
                
                if faq_list:
                    for li in faq_list.find_all("li"):
                        a = li.find("a", href=True)
                        if a:
                            question = a.get_text(strip=True)
                            if question not in seen_questions:
                                link = BASE_URL + a['href']
                                # Get answer via Requests
                                try:
                                    res = requests.get(link, timeout=10)
                                    ans_soup = BeautifulSoup(res.text, "html.parser")
                                    ans_div = ans_soup.find("div", class_="this-info")
                                    answer = ans_div.get_text(separator=" ", strip=True) if ans_div else "No Answer"
                                    
                                    faqs_data.append({"question": question, "answer": answer, "url": link})
                                    seen_questions.add(question)
                                    print(f"Scraped: {question[:50]}...")
                                except:
                                    continue

                # Try to go to next page
                try:
                    next_page_num = page + 1
                    next_btn = driver.find_element(By.XPATH, f'//ul[@id="pager"]//a[text()="{next_page_num}"]')
                    driver.execute_script("arguments[0].click();", next_btn)
                    time.sleep(2)
                except:
                    break # No more pages

        except Exception as e:
            print(f"Finished options or encountered error: {e}")
            break

finally:
    driver.quit()

# Save data
with open("delta_faq.jsonl", "w", encoding="utf-8") as f:
    for entry in faqs_data:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")