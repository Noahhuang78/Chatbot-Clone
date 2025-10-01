import ChatbotIcon from "./ChatbotIcon";
const ChatMessage = ({ chat }) => {
  function renderMessageWithLinks(message) {
  const urlRegex = /(https?:\/\/[^\s]+)/g;

  const parts = message.split(urlRegex); // splits text by URLs

  return parts.map((part, idx) => {
    if (part.match(urlRegex)) {
      part = part.replace(/[.,;!?]+$/, "") //remove fullstop at end of link
      return (
        <a key={idx} href={part} target="_blank" rel="noopener noreferrer">
          {part}
        </a>
      );
    } else {
      return <span key={idx}>{part}</span>;
    }
  });
}
  return (
    !chat.hideInChat && (
    <div
      className={`message ${chat.role === "model" ? "bot" : "user"}-message ${chat.isError ? "error" : ""}`}
    >
        {/* conditional rendering chatbot Icon, if chat.role is model then have chatboticon */}
      {chat.role === "model" && <ChatbotIcon />}    
      <p className="message-text">{renderMessageWithLinks(chat.text)}</p>
    </div>
  )
);
};

export default ChatMessage;
