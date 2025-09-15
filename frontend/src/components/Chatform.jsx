import { useRef } from "react"
const ChatForm = ({ chatHistory, setChatHistory, generateBotResponse}) => {
    const inputRef = useRef();

    const handleFormSubmit = (e) => {
        e.preventDefault()
        const userMessage = inputRef.current.value.trim(); //trim() removes whitespace from both ends of string
        if(!userMessage) return;
        inputRef.current.value = "";
        //add user message to chat history
        setChatHistory((history) => [...history, { role: "user", text: userMessage }])
        // console.log(userMessage)
        // console.log(history)
       
        setTimeout(() => { 
        //Add a "Thinking... " placeholder for bot's response
        setChatHistory((history) => [...history, {role: "model", text: "Thinking..." }]);
        //call function to generate bot response
        generateBotResponse([...chatHistory, { role: "user", text: `Following the chat history provided above and the RAG retrieved context, please address this query: ${userMessage}`}]);
        }, 600)

    }
    return (
        <form action="#" className="chat-form" onSubmit={handleFormSubmit}>
            <input ref={inputRef} type="text" placeholder="Message..." className="message-input" required />
            <button className="material-symbols-rounded">arrow_upward</button>
        </form>
    )
}

export default ChatForm