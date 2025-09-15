import React, { useEffect, useState } from "react";
import ChatbotIcon from "./components/ChatbotIcon";
import ChatForm from "./components/Chatform";
import ChatMessage from "./components/ChatMessage";
import { useRef } from "react"
import { chatbotPrompt } from "./chatbotPrompt";

const App = () => {
  const [chatHistory, setChatHistory] = useState([{
    hideInChat: true,
    role: "model",
    text: chatbotPrompt
  }])
  const [showChatbot, setShowChatbot] = useState(false)

  const chatBodyRef = useRef()

  const generateBotResponse = async(history) => {
    const updateHistory = (text, isError = false) => {  //replaces "Thinking..." with bot's response
      setChatHistory(prev => [...prev.filter(msg => msg.text !== "Thinking..."), {role: "model", text, isError}])
    }
    //Format chat history for API request
    // history = history.map(({role, text}) => ({role, parts: [{text}]}))
    const requestOptions = {
      method: "POST",
      headers: { "Content-Type": "application/json"},
      body: JSON.stringify({contents: history}),
    }
    try{
      //make the API call to get the bot's response
      const response = await fetch("/chatResponse", requestOptions);
      const data = await response.json();
      if(!response.ok) throw new Error(data.error.message || "something went wrong")
      console.log(data)
      //clean and update chat history with bot's response
      const apiResponseText = data.response.replace(/\*\*(.*?)\*\*/g, "$1").trim()
      updateHistory(apiResponseText)
    }catch(error){
      updateHistory(error.message, true)
    }
    console.log(history)
  }

  useEffect(() => {
    chatBodyRef.current.scrollTo({top: chatBodyRef.current.scrollHeight, behavior: "smooth"})
  }, [chatHistory])

  return (
    <div className={`container ${showChatbot ? "show-chatbot" : ""}`}>
      <button onClick={() => { setShowChatbot(prev => !prev)}} id="chatbot-toggler">
        <span className="material-symbols-rounded">mode_comment</span>
        <span className="material-symbols-rounded">close</span>
      </button>
      <div className="chatbot-popup">
        {/* Chatbot Header */}
        <div className="chat-header">
          <div className="header-info">
            <ChatbotIcon />
            <h2 className="logo-text">Chatbot</h2>
          </div>
          <button onClick={() => { setShowChatbot(prev => !prev)}} className="material-symbols-rounded">keyboard_arrow_down</button>
        </div>

        {/* Chatbot Body */}
        <div ref={chatBodyRef} className="chat-body">
          <div className="message bot-message">
            <ChatbotIcon />
            <p className="message-text">Hey there 👋 <br/> How can I help today?</p>
          </div>

          {/* Render the chat history dynamically */}
          {chatHistory.map((chat, index) => (
            <ChatMessage key={index} chat={chat}/>
          ))}
        </div>

        {/* Chatbot Footer */}
        <div className="chat-footer">
          <ChatForm chatHistory={chatHistory} setChatHistory={setChatHistory} generateBotResponse={generateBotResponse}/>
        </div>
      </div>
    </div>
  );
};

export default App;
