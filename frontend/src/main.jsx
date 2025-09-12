import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import ReactDOM  from 'react-dom/client'
import './index.css'
import App from './App.jsx'

createRoot(document.getElementById('root')).render(
  <StrictMode>
    <App />
  </StrictMode>,
)

// -------------  Build as iief.js --------------------
// export function initChatbot( containerId = "chatbot-root"){
//   let container = document.getElementById(containerId)
//   if (!container){
//     container = document.createElement('div')
//     container.id = containerId
//     document.body.appendChild(container)
//   }

//   const root = ReactDOM.createRoot(container)
//   root.render(
//   <StrictMode>
//     <App />
//   </StrictMode>,
//   )
// }

// if (typeof window !== "undefined") {
//   window.MyChatbot = { init: initChatbot() };
// }