import ChatbotIcon from "./ChatbotIcon";
const ChatMessage = ({ chat }) => {
  return (
    !chat.hideInChat && (
    <div
      className={`message ${chat.role === "model" ? "bot" : "user"}-message ${chat.isError ? "error" : ""}`}
    >
        {/* conditional rendering chatbot Icon, if chat.role is model then have chatboticon */}
      {chat.role === "model" && <ChatbotIcon />}    
      <p className="message-text">{chat.text}</p>
    </div>
  )
);
};

export default ChatMessage;
