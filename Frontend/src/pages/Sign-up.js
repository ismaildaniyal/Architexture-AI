/* eslint-disable no-undef */
/* eslint-disable react/jsx-no-undef */
import React, { useState } from "react";
import { useNavigate, Link } from "react-router-dom"; // Import useNavigate from react-router-dom
import InputGroup from "../Components/inputgroup";
import PasswordGroup from "../Components/password";
// import SocialButtons from "./Components/Social";
import axios from "axios";
// import Cookies from "js-cookie"; // Import js-cookie
import "../styles/Sign-up.css";
import Navbar from "../Components/navbar";

// Axios Configuration
axios.defaults.xsrfCookieName = "csrftoken";
axios.defaults.xsrfHeaderName = "X-CSRFToken";
axios.defaults.withCredentials = true;

function Signup() {
  const [username, setUsername] = useState(""); // State for username
  const [email, setEmail] = useState(""); // State for email
  const [password, setPassword] = useState(""); // State for password
  const [errorMessage, setErrorMessage] = useState(""); // To show error messages
  const [isLoading, setIsLoading] = useState(false);
  const navigate = useNavigate(); // Initialize useNavigate hook

  // Handle form submission
   // Validate email format using regex
   const validateEmail = (email) => {
    const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
    return emailRegex.test(email);
  };

  
  const validatePassword = (value) => {
    const passwordRegex =/^(?=.*?[A-Z])(?=.*?[a-z])(?=.*?[0-9])(?=.*?[#?!@$%^&*-]).{8,}$/;
  return passwordRegex.test(value);
  };
  const handleSubmit = async (e) => {
    e.preventDefault(); // Prevent page reload

    // Simple client-side validation to ensure no field is empty
    if (!username || !email || !password) {
      setErrorMessage("Please fill in all fields.");
      return;
    }

    if (!validateEmail(email)) {
      alert("Invalid email format. Please enter a valid email address.");
      return;
    }
  
    if (!validatePassword(password)) {
      alert("Invalid password format. Please have must have Capitial,Small,special,Number.");
      return;
    }

    setIsLoading(true); // Show spinner
    try {
      const response = await axios.post("http://127.0.0.1:8000/api/signup/", {
        username,
        email,
        password,
      });

      if (response.status === 201) {
        // If user created successfully, inform the user to check email
        alert("User Created Successfully. Please check your email to verify your account.");
        navigate("/login"); // Redirect to login page
      }
    } catch (error) {

      if (error.response) {
        if (error.response.status === 300) {
          // Invalid email case
          alert("The email address is invalid. Please enter a valid email.");
          // setErrorMessage("The email address is invalid. Please enter a valid email.");
        } else if (error.response.status === 400) {
          alert("Email already exist")
          // Handle any other server-side errors (e.g., missing fields, etc.)
          // setErrorMessage("An error occurred. Please try again later.");
        } else {
          alert("An unexpected error occurred. Please try again.")
          // setErrorMessage("An unexpected error occurred. Please try again.");
        }
      } else {
        alert("An error occurred. Please try again.")
        // setErrorMessage("An error occurred. Please try again.");
      }
    }finally {
      setIsLoading(false); // Hide spinner
    }
  };

  return (
    <div>
      <Navbar />
      <div className="Signin-page">
        {/* <div className="circle"></div>
        <div className="circle"></div>
        <div className="circle"></div>
        <div className="line"></div>
        <div className="line"></div>
        <div className="line"></div> */}
        <div className="Signin-container">
          <h2>Create New Account</h2>

          {/* Error message display */}
          {errorMessage && <div className="error-message">{errorMessage}</div>}

          <form onSubmit={handleSubmit}  className={isLoading ? "blurred" : ""}>
            {/* Input for Username */}
            <InputGroup
              type="text"
              name="username"
              placeholder="Username"
              required
              value={username}
              onChange={(e) => setUsername(e.target.value)}
            />

            {/* Input for Email */}
            <InputGroup
              type="email"
              name="email"
              placeholder="Email"
              required
              value={email}
              onChange={(e) => setEmail(e.target.value)}
            />

            {/* Input for Password */}
            <PasswordGroup
              type="password"
              name="password"
              placeholder="Password"
              required
              value={password}
              onChange={(e) => setPassword(e.target.value)}
            />

           
            {/* Submit Button */}
            <button type="submit" className="signin-btn" >
              Sign Up
            </button>
            <div className="signup-links">
            <p>Already have an account? <Link to="/login">Login</Link></p>
          </div>
          </form>
          {isLoading && <div className="spinner-overlay">
                <div className="spinner"></div>
                </div>}
              {errorMessage && <p className="error-message">{errorMessage}</p>}
        </div>
      </div>
    </div>
  );
}

export default Signup;
