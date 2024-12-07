import React, { useState, useEffect, useRef } from 'react';
import Cookies from 'js-cookie';
import { Link, useNavigate } from 'react-router-dom';
import '../styles/home.css';
import { FaSun, FaMoon, FaBars, FaTrash } from 'react-icons/fa'; // Importing the hamburger icon
import ProCard from "../Components/card";

function ChatbotUI() {
    const [input, setInput] = useState('');
    const [history, setHistory] = useState({});
    const [currentPromptId, setCurrentPromptId] = useState(1);
    const [showWelcome, setShowWelcome] = useState(true);
    const [username, setUsername] = useState('');
    const [isDarkMode, setIsDarkMode] = useState(false);
    const [isDropdownVisible, setIsDropdownVisible] = useState(false);
    const [selectedPromptId, setSelectedPromptId] = useState(1);
    const [isSidebarVisible, setIsSidebarVisible] = useState(window.innerWidth > 1023);
    const dropdownRef = useRef(null);
    const navigate = useNavigate();
    const messagesEndRef = useRef(null);
    const [windowWidth, setWindowWidth] = useState(window.innerWidth);

    useEffect(() => {
        if (messagesEndRef.current) {
            messagesEndRef.current.scrollIntoView({ behavior: 'smooth' });
        }
    }, [history]);

    useEffect(() => {
        const loggedInUser = Cookies.get('username');
        if (loggedInUser) {
            setUsername(loggedInUser);
        } else {
            setUsername('Guest');
        }
    }, []);

    useEffect(() => {
        const loggedInUser = Cookies.get('username');
        if (!loggedInUser) {
            navigate('/login');
        }
    }, [navigate]);

    const handleLogout = () => {
        Object.keys(Cookies.get()).forEach((cookieName) => {
            Cookies.remove(cookieName);

        });
alert("You have been logged out.");
        navigate("/Login");
    };

    useEffect(() => {
        const handleClickOutside = (event) => {
            if (dropdownRef.current && !dropdownRef.current.contains(event.target)) {
                setIsDropdownVisible(false);
            }
        };

        document.addEventListener('click', handleClickOutside);
        return () => {
            document.removeEventListener('click', handleClickOutside);
        };
    }, []);

    const handleSendPrompt = (mode) => {
        if (input.trim()) {
            const newPrompt = { text: input, mode };
            setHistory((prevHistory) => ({
                ...prevHistory,
                [selectedPromptId]: [...(prevHistory[selectedPromptId] || []), newPrompt],
            }));
            setShowWelcome(false);
            setInput('');
        }
    };

    const handleNewChat = () => {
        if (history[selectedPromptId] && history[selectedPromptId].length > 0) {
            const newPromptId = currentPromptId + 1;
            setCurrentPromptId(newPromptId);
            setSelectedPromptId(newPromptId);
            setHistory((prevHistory) => ({
                ...prevHistory,
                [newPromptId]: [],
            }));
        } else {
            alert("Please add a prompt before starting a new chat.");
        }
    };

    const handleSelectPrompt = (promptId) => {
        setSelectedPromptId(promptId);
    };

    const handleDeletePrompt = (promptId, e) => {
        e.stopPropagation();
        setHistory((prevHistory) => {
            const newHistory = { ...prevHistory };
            delete newHistory[promptId];
            
            // If we're deleting the currently selected prompt, select another one
            if (selectedPromptId === promptId) {
                const remainingIds = Object.keys(newHistory);
                if (remainingIds.length > 0) {
                    setSelectedPromptId(Number(remainingIds[0]));
                }
            }
            
            return newHistory;
        });
    };

    // Handle window resize
    useEffect(() => {
        const handleResize = () => {
            const width = window.innerWidth;
            setWindowWidth(width);
            
            // Update sidebar visibility based on screen size
            if (width > 1023) {
                setIsSidebarVisible(true);
            } else {
                setIsSidebarVisible(false);
            }
        };

        // Add event listener
        window.addEventListener('resize', handleResize);
        
        // Cleanup
        return () => window.removeEventListener('resize', handleResize);
    }, []);

    // Get dynamic styles based on window width
    const getResponsiveStyles = () => {
        if (windowWidth <= 480) {
            return {
                sidebar: {
                    marginTop: "2%",
                    height: "97vh",
                    width: "200px",
                },
                welcome: {
                    marginTop: "20%",
                    marginLeft: "2%",
                    fontSize: "15px",
                },
                chatArea: {
                    marginTop: "14%",
                },
                architextureText: {
                    fontSize: "20px",
                },
                architextureContainer: {
                    marginLeft: "8%",
                },
            };
        } else if (windowWidth <= 768) {
            return {
                // ... styles for tablets
            };
        } else if (windowWidth >= 1440) {
            return {
                // ... styles for large screens
            };
        }
        return {}; // default styles
    };

    const responsiveStyles = getResponsiveStyles();

    return (
        <div className='page'>
            <div className={`chatbot-container ${isDarkMode ? 'light-mode' : 'dark-mode'}`}>
                <div className="top-bar">
                    <div className="architexture-container" style={responsiveStyles.architextureContainer}>
                        <h1 className="architexture-text" style={responsiveStyles.architextureText}>
                            2D - 3D FLOOR PLAN GENERATION
                        </h1>
                    </div>
                    <div className="username-bar">
                        <button className="theme-toggle" onClick={() => setIsDarkMode(!isDarkMode)}>
                            {isDarkMode ? <FaMoon style={{ color: '#3536377d' }} /> : <FaSun />}
                        </button>
                        <button className="hamburger-icon" onClick={() => setIsSidebarVisible(!isSidebarVisible)}>
                            <FaBars />
                        </button>
                        <span
                            className="username"
                            onClick={(e) => {
                                e.stopPropagation();
                                setIsDropdownVisible(!isDropdownVisible);
                            }}
                        >
                            {username.substring(0, 2)}
                        </span>
                        {isDropdownVisible && (
                            <div ref={dropdownRef} className="dropdown">
                                <p className="dropdown-item">👋 Hey, {username}</p>
                                <Link to="/Login" className="logout-button" onClick={handleLogout}>
                                    Logout
                                </Link>
                            </div>
                            
                        )}
                        {/* Hamburger Menu Icon */}
                    </div>
                </div>
                <div 
                    className={`sidebar-overlay ${isSidebarVisible ? 'show' : ''}`} 
                    onClick={() => setIsSidebarVisible(false)}
                ></div>

                {/* Sidebar */}
                <div 
                    className={`sidebar ${isSidebarVisible ? 'show' : 'hide'}`}
                    style={{ 
                        transform: isSidebarVisible ? 'translateX(0)' : 'translateX(-110%)',
                        display: 'block',
                        position: windowWidth <= 1023 ? 'fixed' : 'relative',
                        ...responsiveStyles.sidebar
                    }}
                >
                    <h2>Architexture AI</h2>
                    <span>_______________________________</span>
                    <button className="new-chat-button" onClick={handleNewChat}>
                        New Chat
                    </button>
                    <div className="history">
                        {Object.keys(history).map((promptId) => (
                            <div
                                key={promptId}
                                className={`history-item ${selectedPromptId === Number(promptId) ? 'selected' : ''}`}
                                onClick={() => handleSelectPrompt(Number(promptId))}
                            >
                                <span>Prompt {promptId}</span>
                                <button 
                                    className="delete-prompt-btn"
                                    onClick={(e) => handleDeletePrompt(Number(promptId), e)}
                                >
                                    <FaTrash size={14} />
                                </button>
                            </div>
                        ))}
                    </div>
                    <div className='sidebar-bottom-card' >
                        <ProCard />
                    </div>
                  
                </div>

                <div className="chat-area" style={responsiveStyles.chatArea}>
                    <div className="messages">
                        {showWelcome && (
                            <div className="welcome" style={responsiveStyles.welcome}>
                                <div className="welcome1">
                                    <h1>Hello,</h1>
                                    <div className="welcome-username">
                                        <h1 className="color-change">
                                            {username.split("").map((char, index) => (
                                                <span key={index}>{char}</span>
                                            ))}
                                        </h1>
                                    </div>
                                </div>
                                <h1>What do you want to create</h1>
                            </div>
                        )}
                        {history[selectedPromptId] && history[selectedPromptId].map((prompt, index) => (
                            <div key={index} className="message">
                                <div className="user-message">
                                    <span className="user-icon">👤</span>
                                    <span className="prompt-text">{prompt.text}</span>
                                </div>
                                <div className="response">
                                    <span className="response-icon">🤖</span>
                                    <span className="response-text">Hello, thank you!</span>
                                </div>
                            </div>
                        ))}
                        <div ref={messagesEndRef}></div>
                    </div>

                    <div className="input-area">
                        <textarea
                            className="input-box"
                            value={input}
                            onChange={(e) => setInput(e.target.value)}
                            onKeyDown={(e) => {
                                if (e.key === 'Enter' && e.shiftKey) {
                                    setInput(input + '\n');
                                } else if (e.key === 'Enter') {
                                    handleSendPrompt("2D");
                                    e.preventDefault();
                                }
                            }}
                            placeholder="Type your message here..."
                            rows={1}
                        />
                        <button className="mode-button" onClick={() => handleSendPrompt("2D")}>2D</button>
                        <button className="mode-button" onClick={() => handleSendPrompt("3D")}>3D</button>
                    </div>
                </div>
            </div>
        </div>
    );
}

export default ChatbotUI;
