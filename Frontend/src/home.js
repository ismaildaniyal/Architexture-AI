import React, { useState,useEffect } from 'react';
import './home.css';

function ChatbotUI() {
    const [input, setInput] = useState('');
    const [history, setHistory] = useState([]);
    const [displayedPrompt, setDisplayedPrompt] = useState('');
    const [showWelcome, setShowWelcome] = useState(true); // State to track welcome visibility
    const [username, setUsername] = useState(''); // State for the logged-in username


    useEffect(() => {
        const loggedInUser = localStorage.getItem('username'); // Example fetching from localStorage, change as per your app logic
        if (loggedInUser) {
            setUsername(loggedInUser); // Set the username state to the logged-in user's name
        } else {
            setUsername('Guest'); // Default if no user is logged in
        }
    }, []); // Run only once when the component mounts
    

    // Handle sending messages when clicking 2D or 3D buttons
    const handleSendPrompt = (mode) => {
        if (input.trim()) {
            const newPrompt = { text: input, mode };
            setHistory([...history, newPrompt]); // Add to history
            setDisplayedPrompt(newPrompt.text); // Display in output area
            setShowWelcome(false); // Hide the welcome message when a prompt is sent
            setInput(''); // Clear input

        }
    };

    // Handle input changes
    const handleInputChange = (e) => {
        setInput(e.target.value);
    };

    // Handle keydown for line breaks (Shift + Enter)
    const handleKeyDown = (e) => {
        if (e.key === 'Enter' && e.shiftKey) {
            setInput(input + '\n'); // Add line break
        } else if (e.key === 'Enter') {
            handleSendPrompt("2D"); // Default to 2D on Enter key
        }
    };

    // Handle resetting the chat history
    const handleNewChat = () => {
        setInput(''); // Clear the input
        setDisplayedPrompt(''); // Reset displayed prompt
        setShowWelcome(true); // Show welcome message again
        // setHistory([]); // Clear the chat history
    };

    return (
        <div className="chatbot-container">
            {/* Top bar with username and "Architexture" text */}
            <div className="top-bar">
                <div className="architexture-container">
                    <h1 className="architexture-text">ARCHITEXTURE</h1>
                    <p className="standard-text">Standard 3.5</p>
                </div>
                <span className="username">{username}</span>
            </div>
            {/* Left sidebar with prompt history */}
            <div className="sidebar">
                <button className="new-chat-button" onClick={handleNewChat}>
                    ➕ New Chat
                </button>
                <div className="history">
                    {history.map((prompt, index) => (
                        <div key={index} className="history-item">
                            {prompt.text}
                        </div>
                    ))}
                </div>
            </div>

            {/* Main chat area */}
            <div className="chat-area">
                <div className="messages">
                    {showWelcome && (
                        <div className='welcome'>
                            <div className='welcome1'>
                                <h1>Hello,</h1>
                                <div className='welcome-username'>
                                    <h1 className="color-change"> {username.split("").map((char, index) => (
                                        <span key={index}>{char}</span>
                                    ))}</h1>
                                </div>
                            </div>
                            <h1>What do you want to create</h1>
                        </div>
                    )}

                    {displayedPrompt && (
                        <div className="user-prompt">
                            <span className="user-icon">👤</span>
                            <span className="prompt-text">{displayedPrompt}</span>
                        </div>
                    )}
                    {/* Display chatbot response */}
                    {history
                        .filter(item => item.mode === 'response')
                        .map((response, index) => (
                            <div key={index} className="chatbot-response">
                                <span className="chatbot-icon">🤖</span>
                                <span className="response-text">{response.text}</span>
                            </div>
                    ))}
                </div>

                {/* Input area at the bottom */}
                <div className="input-area">
                    <textarea
                        className="input-box"
                        value={input}
                        onChange={handleInputChange}
                        onKeyDown={handleKeyDown}
                        placeholder="Type your message here..."
                        rows={1} // Ensures it expands as needed
                    />
                    <button className="mode-button" onClick={() => handleSendPrompt("2D")}>2D</button>
                    <button className="mode-button" onClick={() => handleSendPrompt("3D")}>3D</button>
                </div>
            </div>
        </div>
    );
}

export default ChatbotUI;