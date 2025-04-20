import React, { useState, useEffect, useRef } from 'react';
import Cookies from 'js-cookie';
import { Link, useNavigate } from 'react-router-dom';
import '../styles/home.css';
import { FaSun, FaMoon, FaBars, FaTrash } from 'react-icons/fa'; // Importing the hamburger icon
import ProCard from "../Components/card";
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome';
import { faHouseChimney } from '@fortawesome/free-solid-svg-icons';
import { faUser } from '@fortawesome/free-solid-svg-icons';
function HomePage() {
    const [input, setInput] = useState('');
    const [history, setHistory] = useState({ 1: [] });
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

    const [isOpen, setIsOpen] = useState(false);
    const [selectedImage, setSelectedImage] = useState(null); // Store clicked image

    const openModal = (imageUrl) => {
        setSelectedImage(imageUrl); // Save the clicked image
        setIsOpen(true);
    };

    const closeModal = () => {
        setIsOpen(false);
        setSelectedImage(null); // Clear image when closing
    };

    // Function to download the image
    const downloadImage = () => {
        if (!selectedImage) return;
        const link = document.createElement("a");
        link.href = selectedImage;
        link.download = "House_Plan.png"; // Default filename
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
    };




    
    useEffect(() => {
        const fetchChatHistory = async () => {
            try {
                const userEmail = Cookies.get('email'); // Get email from cookies
                if (!userEmail) throw new Error("User email not found. Please login again.");
    
                const response = await fetch(`http://127.0.0.1:8000/api/retrive-data?email=${userEmail}`, {
                    method: 'GET',
                    headers: { 'Content-Type': 'application/json' },
                    credentials: 'include',
                });
    
                if (!response.ok) {
                    throw new Error("Failed to fetch chat history.");
                }
    
                const data = await response.json();
                console.log("Fetched chat history:", data);
    
                if (!data.all_chats || data.all_chats.length === 0) {
                    // If no chats exist, show the welcome screen
                    setShowWelcome(true);
                    return;
                }
    
                // Sort chats by created_at (Descending Order)
                const sortedChats = data.all_chats.sort((a, b) => new Date(b.created_at) - new Date(a.created_at));
    
                // Initialize history object with chat IDs
                const newHistory = {};
                sortedChats.forEach(chat => {
                    newHistory[chat.chat_id] = [];
                });
    
                // Set history and hide the welcome screen
                setHistory(newHistory);
                setShowWelcome(false);
    
                // Set the latest chat ID
                const latestChatId = data.latest_chat.chat_id;
                setSelectedPromptId(latestChatId);
    
                setHistory((prevHistory) => ({
                    ...prevHistory,
                    [latestChatId]: data.latest_chat.prompts
                        .sort((a, b) => a.id - b.id) // Sort by ID in ascending order
                        .map((prompt) => {
                            const imageName = prompt.image_path ? prompt.image_path.split('\\').pop() : null;
                            console.log('Image name:', imageName);
                            return {
                                text: prompt.prompt_text,
                                output_text: prompt.output_text,
                                image: imageName ? `http://127.0.0.1:8000/api/get-image/${imageName}` : null, // Fetch from API
                                success: !!prompt.image_path, // Success if image path exists
                                message: prompt.image_path ? null : prompt.output_text || "No valid image available.",
                            };
                        }),
                }));
    
            } catch (error) {
                console.error("Error fetching chat history:", error);
            }
        };
    
        fetchChatHistory();
    }, []);
    
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
    const [loading, setLoading] = useState(false);

    const handleSendPrompt = async (mode) => {
        if (input.trim()) {
            setLoading(true); // Start loading
            setShowWelcome(false); // Hide the welcome screen
            const currentInput = input; 
            const newPrompt = { text: input, mode };
    
            // Add the input message with a loading state to history
            setHistory((prevHistory) => ({
                ...prevHistory,
                [selectedPromptId]: [...(prevHistory[selectedPromptId] || []), { ...newPrompt, loading: true }],
            }));
            setInput("");

            try {
                const userEmail = Cookies.get('email'); // Get email from cookie
                // Debug logs
                console.log('Sending request with data:', {
                    input,
                    mode,
                    email: userEmail,
                    promptId: selectedPromptId,
                });
                
                if (!userEmail) {
                    throw new Error('User email not found. Please login again.');
                }

                if (!selectedPromptId) {
                    throw new Error('No prompt ID selected.');
                }

                const requestData = {
                    input: input,
                    mode: mode,
                    email: userEmail,
                    promptId: selectedPromptId  // Changed from selectedPromptId to promptId
                };
                

                console.log('Request body:', JSON.stringify(requestData));

                const response = await fetch('http://127.0.0.1:8000/api/process-houseplan/', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    credentials: 'include',  // Include cookies in the request
                    body: JSON.stringify(requestData),
                });
    
                let newResponse = {}; // To store the response
    
                if (response.ok) {
                    // Convert response to Blob for image handling
                const blob = await response.blob();
                const imageUrl = URL.createObjectURL(blob);
                console.log('Image received:', imageUrl);

                newResponse = {
                    image: imageUrl,
                    success: true,
                    text: input,
                };
                } else {
                    const errorData = await response.json();
                    console.error('Error response:', errorData);
                    newResponse = {
                        message: errorData.error || 'An error occurred.',
                        success: false,
                        text: input,
                    };
                }
    
                // Update history with the response
                setHistory((prevHistory) => ({
                    ...prevHistory,
                    [selectedPromptId]: prevHistory[selectedPromptId].map((prompt) =>
                        prompt.text === currentInput ? newResponse : prompt
                    ),
                }));
            } catch (error) {
                console.error('Request error:', error);
                const newResponse = {
                    message: error.message || 'An error occurred. Please try again.',
                    success: false,
                    text: input,
                };
    
                // Update history with the error message
                setHistory((prevHistory) => ({
                    ...prevHistory,
                    [selectedPromptId]: prevHistory[selectedPromptId].map((prompt) =>
                        prompt.text === currentInput ? newResponse : prompt
                    ),
                }));
            } finally {
                setLoading(false); // Stop loading
            }
        }
    };
    
    
    const handleSendPrompt1 = async (mode) => {
        if (input.trim()) {
            setLoading(true); // Start loading
            setShowWelcome(false); // Hide the welcome screen
            const currentInput = input; // Capture current input

            const newPrompt = { text: input, mode };
    
            // Add the input message with a loading state to history
            setHistory((prevHistory) => ({
                ...prevHistory,
                [selectedPromptId]: [...(prevHistory[selectedPromptId] || []), { ...newPrompt, loading: true }],
            }));
            setInput("");

            try {
                const userEmail = Cookies.get('email'); // Get email from cookie
                // Debug logs
                console.log('Sending request with data:', {
                    input,
                    email: userEmail,
                    promptId: selectedPromptId,
                });
                
                if (!userEmail) {
                    throw new Error('User email not found. Please login again.');
                }

                if (!selectedPromptId) {
                    throw new Error('No prompt ID selected.');
                }

                const requestData = {
                    input: input,
                    mode: mode,
                    email: userEmail,
                    promptId: selectedPromptId  // Changed from selectedPromptId to promptId
                };
                

                console.log('Request body:', JSON.stringify(requestData));

                const response = await fetch('http://127.0.0.1:8000/api/3D-Design/', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    credentials: 'include',  // Include cookies in the request
                    body: JSON.stringify(requestData),
                });
            
                let newResponse = {}; // To store the response
    
                if (response.ok) {
                    // Convert response to Blob for image handling
                const blob = await response.blob();
                const imageUrl = URL.createObjectURL(blob);
                console.log('Image received:', imageUrl);

                newResponse = {
                    image: imageUrl,
                    success: true,
                    text: input,
                };
                } else {
                    const errorData = await response.json();
                    console.error('Error response:', errorData);
                    newResponse = {
                        message: errorData.error || 'An error occurred.',
                        success: false,
                        text: input,
                    };
                }
    
                // Update history with the response
                setHistory((prevHistory) => ({
                    ...prevHistory,
                    [selectedPromptId]: prevHistory[selectedPromptId].map((prompt) =>
                        prompt.text === currentInput ? newResponse : prompt
                    ),
                }));
            } catch (error) {
                console.error('Request error:', error);
                const newResponse = {
                    message: error.message || 'An error occurred. Please try again.',
                    success: false,
                    text: input,
                };
    
                // Update history with the error message
                setHistory((prevHistory) => ({
                    ...prevHistory,
                    [selectedPromptId]: prevHistory[selectedPromptId].map((prompt) =>
                        prompt.text === currentInput ? newResponse : prompt
                    ),
                }));
            } finally {
                setLoading(false); // Stop loading
            }
        }
    };
    
    
    
    

    const handleNewChat = () => {
        if (history[selectedPromptId] && history[selectedPromptId].length > 0) {
            const newPromptId = currentPromptId + 1;
            setCurrentPromptId(newPromptId);
            setSelectedPromptId(newPromptId);
            
            // Add new chat at the beginning (top of stack)
            setHistory((prevHistory) => ({
                [newPromptId]: [],
                ...prevHistory,
            }));
            // Only show welcome for the new empty chat
            setShowWelcome(true);
        } else {
            alert("Please add a prompt before starting a new chat.");
        }
    };

    const handleSelectPrompt = async (promptId) => {
        setSelectedPromptId(promptId);
        
        try {

            const userEmail = Cookies.get('email');
            const response = await fetch(
                `http://127.0.0.1:8000/api/retrive-data?email=${userEmail}&chat_id=${promptId}`
            );
            const data = await response.json();
    
            if (data && Array.isArray(data)) {
                setHistory((prevHistory) => ({
                    ...prevHistory,
                    [promptId]: data
                        .sort((a, b) => a.id - b.id) // Sort by ID in ascending order
                        .map((prompt) => {
                            const imageName = prompt.image_path ? prompt.image_path.split('\\').pop() : null;
                            console.log('Image name:', imageName);
                            return {
                                text: prompt.prompt_text,
                                output_text: prompt.output_text,
                                image: imageName ? `http://127.0.0.1:8000/api/get-image/${imageName}` : null, // Fetch from API
                                success: !!prompt.image_path, // Success if image path exists
                                message: prompt.image_path ? null : prompt.output_text || "No valid image available.",
                            };
                        }),
                }));
            }
        } catch (error) {
            console.error("Error fetching prompt data:", error);
        }
    
        // Show welcome message only if the selected chat has no history
        setShowWelcome(history[promptId]?.length === 0);
    };
    

    const handleDeletePrompt = async (promptId, e) => {
        e.stopPropagation();
    
        // Get email from cookies
        const userEmail = Cookies.get("email"); // Ensure 'email' is the correct cookie name
    
        if (!userEmail) {
            alert("User email not found. Please log in again.");
            return;
        }
    
        try {
            const response = await fetch(`http://127.0.0.1:8000/api/delete-chat?email=${userEmail}&chat_id=${promptId}`, {
                method: "DELETE",
                headers: {
                    "Content-Type": "application/json",
                },
            });
    
            console.log("Delete chat response:", response);
            
            if (response.status === 200) {
                setHistory((prevHistory) => {
                    const newHistory = { ...prevHistory };
                    delete newHistory[promptId]; // Only delete the selected prompt without renumbering
    
                    // If the deleted prompt was the selected one, select another prompt
                    if (selectedPromptId === promptId) {
                        const remainingIds = Object.keys(newHistory).map(Number);
                        if (remainingIds.length > 0) {
                            setSelectedPromptId(Math.min(...remainingIds)); // Select the smallest remaining ID
                        }
                    }
    
                    // Update currentPromptId to the highest available ID
                    const maxId = Math.max(...Object.keys(newHistory).map(Number), 0);
                    setCurrentPromptId(maxId);
    
                    return newHistory; // Return updated history without renumbering
                });
            } else {
                alert("Failed to delete chat. Please try again.");
            }
        } catch (error) {
            console.error("Error deleting chat:", error);
            alert("An error occurred while deleting the chat.");
        }
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
                        {Object.keys(history)
                            .sort((a, b) => b - a) // Sort in descending order to show newest first
                            .map((promptId) => (
                                <div
                                    key={promptId}
                                    className={`history-item ${selectedPromptId === Number(promptId) ? 'selected' : ''}`}
                                    style={{
                                        backgroundColor: selectedPromptId === Number(promptId) ? '#6a6af0' : 'transparent',
                                        color: selectedPromptId === Number(promptId) ? 'white' : 'inherit'
                                    }}
                                    onClick={() => handleSelectPrompt(Number(promptId))}
                                >
                                    <span>Prompt </span>
                                    <button 
                                        className="delete-prompt-btn"
                                        onClick={(e) => handleDeletePrompt(Number(promptId), e)}
                                        style={{
                                            color: selectedPromptId === Number(promptId) ? 'white' : '#ff6347'
                                        }}
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
                        {(showWelcome && (!history[selectedPromptId] || history[selectedPromptId].length === 0)) && (
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
                                <h1>Bring your vision to life with 2D & 3D plans</h1>
                            </div>
                        )}
                        {history[selectedPromptId] && history[selectedPromptId].map((prompt, index) => (
                            <div key={index} className="message">
                                <div className="user-message">
                                    <span className="user-icon"><FontAwesomeIcon icon={faUser}   style={{ color: '#1c7ed6' }}/></span>
                                    <span className="prompt-text">{prompt.text}</span>
                                </div>
                        {/* Container for chat content and spinner overlay */}
                        <div className="response">
                                <span className="response-icon"><FontAwesomeIcon icon={faHouseChimney} style={{ color: '#1c7ed6' }} /> Architexture AI</span>

                                {/* Check if the request is loading */}
                                {prompt.loading ? (
                                    <div
                                        style={{
                                            position: "relative",
                                            width: "300px",
                                            height: "300px",
                                            border: "1px solid black",
                                            marginTop: "10px",
                                            background: "white",
                                        }}
                                        className="blurred-container"  // Apply blur effect only to this container
                                    >
                                        {/* Spinner Overlay */}
                                        <div className="spinner-overlay">
                                            <div className="spinner"></div>
                                        </div>
                                    </div>
                                ) : prompt.success ? (
                                    <div className="svg-container">
                                    {prompt.image ? (
                                        <>
                                            {/* Clickable Image */}
                                            <img
                                                src={prompt.image}
                                                alt="Generated House Plan"
                                                style={{ width: "300px", height: "300px", border: "1px solid black", background: "white", cursor: "pointer" }}
                                                onClick={() => openModal(prompt.image)} // Pass the clicked image
                                            />
                        
                                            {/* Modal Popup */}
                                            {isOpen && selectedImage && (
                                                <div className="modal-overlay" onClick={closeModal}>
                                                    <div className="modal-content" onClick={(e) => e.stopPropagation()}>
                                                        <span className="close-btn" onClick={closeModal}>&times;</span>
                                                        <img src={selectedImage} alt="Full View" className="modal-image" />
                                                       
                                                    </div>
                                                </div>
                                            )}
                                        </>
                                    ) : (
                                        <p>Loading image...</p>
                                    )}
                        
                                    {/* CSS for Modal */}
                                    <style jsx>{`
                                        .modal-overlay {
                                            position: fixed;
                                            top: 0;
                                            left: 0;
                                            width: 100%;
                                            height: 100%;
                                            background: rgba(0, 0, 0, 0.5);
                                            backdrop-filter: blur(10px);
                                            display: flex;
                                            align-items: center;
                                            justify-content: center;
                                            z-index: 1000;
                                            animation: fadeIn 0.3s ease-in-out;
                                        }
                        
                                        .modal-content {
                                            position: relative;
                                            background: white;
                                            padding: 20px;
                                            border-radius: 10px;
                                            max-width: 90%;
                                            max-height: 90%;
                                            display: flex;
                                            flex-direction: column;
                                            align-items: center;
                                            box-shadow: 0px 4px 10px rgba(0, 0, 0, 0.2);
                                            animation: scaleUp 0.3s ease-in-out;
                                        }
                        
                                        .modal-image {
                                            max-width: 100%;
                                            max-height: 70vh;
                                            border-radius: 8px;
                                        }
                        
                                        .close-btn {
                                            position: absolute;
                                            top: 10px;
                                            right: 15px;
                                            font-size: 24px;
                                            cursor: pointer;
                                            color: black;
                                            font-weight: bold;
                                            transition: transform 0.2s;
                                        }
                        
                                        .close-btn:hover {
                                            transform: scale(1.2);
                                        }
                        
                                        .download-btn {
                                            margin-top: 15px;
                                            padding: 10px 20px;
                                            background: #007bff;
                                            color: white;
                                            border: none;
                                            border-radius: 5px;
                                            cursor: pointer;
                                            font-size: 16px;
                                            transition: background 0.3s;
                                        }
                        
                                        .download-btn:hover {
                                            background: #0056b3;
                                        }
                        
                                        @keyframes fadeIn {
                                            from { opacity: 0; }
                                            to { opacity: 1; }
                                        }
                        
                                        @keyframes scaleUp {
                                            from { transform: scale(0.8); opacity: 0; }
                                            to { transform: scale(1); opacity: 1; }
                                        }
                                    `}</style>
                                </div>
                                    
                                ) : (
                                    <span className="response-text">
                                        {prompt.message || 'Error generating predictions'}
                                    </span>
                                )}
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
    disabled={loading} // Disable input while loading
/>
<button className="mode-button"  onClick={() => handleSendPrompt("2D")} disabled={loading}>
    2D
</button>
<button className="mode-button"  onClick={()=> handleSendPrompt1("3D")} disabled={loading}>
    3D
</button>

                    </div>
                </div>
            </div>
        </div>
    );
}

export default HomePage;