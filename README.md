## Chatbot
- Chatbot that scrapes data from deltaww FAQ as a pseudo knowledge base and uses Chroma vector DB for RAG.
- RAG pipeline includes retrieval of pdf images based on metadata path. (Feature not implemented here, only in Capstone DeltaCare360)
- Industrial Capstone Website here: https://deltaelectronicscapstone.netlify.app/

## Evaluation
Embedding model and vector evaluation comparison

## Try the Chatbot here:
https://chatbot-clone-259334185265.asia-southeast1.run.app/


Some FAQs to ask:
What is Time of Flight (ToF) and what is it used for?
What is image fusion and color fusion, and where are they used in industrial automation?
What is the difference between global shutter and rolling shutter cameras?
Does Delta have a PLC that supports the SAE J1939 protocol?
Does Delta offer products that support MQTT communication?

FAQ questions scraped from here:
https://www.deltaww.com/en-US/service-support/faq
(note: Couple of new questions have not been added by the scraper due to change in UI of the webpage)
