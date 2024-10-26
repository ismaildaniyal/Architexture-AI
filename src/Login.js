// src/Login.js
import React, { useState } from "react";
import { useNavigate } from 'react-router-dom';
import "./Login.css";  // Import the styling
import { FaEye, FaEyeSlash } from "react-icons/fa";

function Login() {
    const [passwordVisible, setPasswordVisible] = useState(false);
   
  
    const togglePasswordVisibility = () => {
      setPasswordVisible(!passwordVisible);
    }; 



    const navigate = useNavigate();

  const handleSignupClick = () => {
    navigate('/signup'); // Redirect to the Signup page
  };

  

  return (
    <div className="login-page">
         <div className="circle"></div>
  <div className="circle"></div>
  <div className="circle"></div>
  <div className="line"></div>
  <div className="line"></div>
  <div className="line"></div>
      <div className="login-container">
        <h2>Login Here</h2>
        <form>
          <div className="input-group">
            <label>Username</label>
            <input type="text" placeholder="Email or Username"  required />
          </div>
          <div className="input-group password-group">
            <label>Password</label>
            <input
              type={passwordVisible ? "text" : "password"}
              placeholder="Password" 
              required
            />
            <span className="eye-icon" onClick={togglePasswordVisibility}>
              {passwordVisible ? <FaEyeSlash /> : <FaEye />}
            </span>
          </div>
          <button type="submit" className="login-btn">Log In</button>
        </form>
        {/* Forgot password and sign-up links */}
        <div className="login-links">
          <a href="" className="forgot-password">Forgot Password?</a>
          <a href="/Sign-in" className="sign-up" onClick={handleSignupClick}>Sign Up</a>
        </div>
        <div className="social-login">
          <button className="social-btn google-btn">Google</button>
          <button className="social-btn facebook-btn">Facebook</button>
        </div>
      </div>
    </div>
  );
}

export default Login;
