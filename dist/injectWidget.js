(function injectStyles(
  hrefs = [
    "https://euphonious-rabanadas-857c44.netlify.app/chatbot.css",
    "https://fonts.googleapis.com/css2?family=Material+Symbols+Rounded:opsz,wght,FILL,GRAD@48,400,1,0",
  ]
) {
  const links = hrefs.map((href) => {
    const link = document.createElement("link");
    link.rel = "stylesheet";
    link.href = href;
    return link;
  });
  links.forEach((link) => document.head.appendChild(link));
})();

(function injectScript() {
  const script = document.createElement("script");
  script.src =
    "https://euphonious-rabanadas-857c44.netlify.app/chatbot.iife.js";
  script.defer = true; //script only runs after raw html is parsed to DOM  e.g <html> <body><div></div> <script></script></body> </html>
  script.onload = () => {
    window.MyChatbot.initChatbot();
  };
  document.head.appendChild(script); //set all settings for script then append it to
})();
