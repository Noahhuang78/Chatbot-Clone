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
                                
// -------------  Build as iife.js --------------------immediately invoked function expression. iife --> (function load(){console.log("loaded")})() //loaded, load() --> this cannot call the iife. executed and not defined.
// export function initChatbot( containerId = "chatbot-root"){       //but iife.js is not a file that executes an immediate function call (beware of this common confusion), main use is its nature of private scoping

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
//       //window is the global object in browsers. window.foo = 123, <in another script> console.log(foo) //123
// if (typeof window !== "undefined") {   //type of window = "object" in browser, but "undefined" in Node.js/non-browser env
//   window.MyChatbot = { init: initChatbot };  //ensures initChatbot() runs if in browser, otherwise if node.js env it won't run but also won't crash the whole frontend
// }