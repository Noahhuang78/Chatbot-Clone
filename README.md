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


## Injecting the Chatbot Widget
- paste this inside your <body> tag in html page: <script src="https://chatbotwidgetbase.netlify.app/injectWidget.js"></script>
- run your html page, the widget should be displayed on the bottom right corner of your page.

## Building the Widget
- comment out createRoot(document.getElementById('root')).render(
  <StrictMode>
    <App />
  </StrictMode>,
) in main.jsx 
- uncomment the Build as iife.js
- npm run build:lib
- drag and drop only the chatbot.iife.js from frontend/dist into root dist folder (don't replace chatbot.css. If you did, delete the *{} and body{} CSS elements otherwise it will override the styling of the html page you are embedding this widget into)
- deploy the root dist folder on netlify
- paste this inside your <body> tag of your html page: <script src="https://<your-netlify-url>/injectWidget.js"></script>
- done! Chatbot widget should be displayed and functioning on your html page!