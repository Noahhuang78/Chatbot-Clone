## Disclaimer & Terms of Use

This chatbot is an independent personal project created for educational and demonstration purposes only. It is not affiliated with, endorsed by, or officially connected to Delta Electronics in any way.

The chatbot may generate incorrect, incomplete, outdated, or misleading responses. Information provided by the chatbot should not be considered official customer support, professional advice, or contractual information.

Users should always verify important details directly through official Delta Electronics channels.

By using this chatbot, you acknowledge and agree that:

- responses are generated automatically and may contain errors
- the creator of this project is not liable for any losses, damages, misunderstandings, or inconveniences resulting from use of the chatbot
- this project is intended solely for non-commercial, portfolio, educational, and experimental purposes

If you require official assistance, please contact Delta Electronics directly through their official website or customer support channels.

## About the Chatbot (This is NOT the industrial Capstone Project)

- A simple chatbot that uses scraped data from deltaww FAQ as a pseudo knowledge base and uses Chroma vector DB for RAG.
- RAG pipeline includes retrieval of pdf images based on metadata path. (Feature not implemented here, only in Capstone DeltaCare360)
- If you are curious to learn more about the actual Industrial Capstone Project (DeltaCare360), You can look at the website here: https://deltaelectronicscapstone.netlify.app/

## Evaluation

Embedding model and vector evaluation comparison

## Try the Chatbot here:

https://chatbot-clone-259334185265.asia-southeast1.run.app/

Some FAQs to ask:

- What is Time of Flight (ToF) and what is it used for?
- What is image fusion and color fusion, and where are they used in industrial automation?
- What is the difference between global shutter and rolling shutter cameras?
- Does Delta have a PLC that supports the SAE J1939 protocol?
- Does Delta offer products that support MQTT communication?

FAQ questions scraped from here:
https://www.deltaww.com/en-US/service-support/faq
(note: Couple of new questions have not been added by the scraper due to change in UI of the webpage)

## Injecting the Chatbot Widget

- paste this inside your <body> tag in html page: <script src="https://chatbotwidgetbase.netlify.app/injectWidget.js"></script>
- run your html page, the widget should be displayed on the bottom right corner of your page.
- (note: box-styling of body has to be in border-box)

## Building the Widget

- comment out createRoot(document.getElementById('root')).render(
  <StrictMode>
  <App />
  </StrictMode>,
  ) in main.jsx
- uncomment the ---Build as iife.js--- section
- npm run build:lib
- drag and drop only the chatbot.iife.js from frontend/dist into root dist folder (don't replace chatbot.css. If you did, replace the _{...} with #chatbot-root _ {
  box-sizing: border-box;} and delete body{...} CSS elements otherwise it will override the styling of the html page you are embedding this widget into. )
- deploy the root dist folder on netlify
- paste this inside your <body> tag of your html page: <script src="https://<your-netlify-url>/injectWidget.js"></script>
- done! Chatbot widget should be displayed and functioning on your html page!
