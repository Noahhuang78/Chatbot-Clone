import os
from celery import Celery
from add_FAQ import scrap_new
from faq_module import  load_faq, embed_faq
redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
celery = Celery("tasks", broker=redis_url, backend=redis_url)

@celery.task
def scrape_faqs():
    scrap_new()
    return "scrape finished"

@celery.task
def process_faqs(scrape_result):
    print("Scrape Result:" + "\n" + scrape_result)
    load_faq()
    embed_faq()
    return "new faqs processed!"