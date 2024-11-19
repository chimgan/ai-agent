import React, { useState } from 'react';
import axios from 'axios';
import './App.css';

function App() {
  const [messages, setMessages] = useState([]);
  const [userInput, setUserInput] = useState('');

  const sendMessage = async () => {
    if (userInput.trim() === '') return;

    const userMessage = { sender: 'user', text: userInput };
    setMessages((prevMessages) => [...prevMessages, userMessage]);

    try {
      const response = await axios.post('http://localhost:5000/chat', {
        message: userInput,
      });

      const botMessage = {
        sender: 'agent',
        text: response.data.response,
        accuracy: response.data.accuracy.toFixed(2),
      };
      setMessages((prevMessages) => [...prevMessages, botMessage]);
    } catch (error) {
      console.error(error);
      const errorMessage = {
        sender: 'agent',
        text: "I'm sorry, something went wrong.",
      };
      setMessages((prevMessages) => [...prevMessages, errorMessage]);
    }

    setUserInput('');
  };

  return (
    <div className="App">
      <h1>AI Support Agent</h1>
      <div className="chat-window">
        {messages.map((msg, index) => (
          <div
            key={index}
            className={msg.sender === 'user' ? 'user-message' : 'agent-message'}
          >
            <strong>{msg.sender === 'user' ? 'You' : 'Agent'}:</strong> {msg.text}
            {msg.sender === 'agent' && (
              <div className="accuracy">Confidence: {msg.accuracy}%</div>
            )}
          </div>
        ))}
      </div>
      <div className="input-area">
        <input
          type="text"
          value={userInput}
          onChange={(e) => setUserInput(e.target.value)}
          placeholder="Type your message here..."
          onKeyPress={(e) => {
            if (e.key === 'Enter') sendMessage();
          }}
        />
        <button onClick={sendMessage}>Send</button>
      </div>
    </div>
  );
}

export default App;
