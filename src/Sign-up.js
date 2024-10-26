// src/Signin.js
import React, { useState } from "react";
import { useNavigate } from 'react-router-dom';
import "./Sign-up.css";  // Import the styling (or reuse Login.css)
import { FaEye, FaEyeSlash } from "react-icons/fa";




function Signin() {
    const [passwordVisible, setPasswordVisible] = useState(false);
  
    const togglePasswordVisibility = () => {
      setPasswordVisible(!passwordVisible);
    }; 

    const navigate = useNavigate();

  const handleLoginClick = () => {
    navigate('/Login'); // Redirect to the Login page
  };

  

  return (
    <div className="Signin-page">
              <div className="circle"></div>
  <div className="circle"></div>
  <div className="circle"></div>
  <div className="line"></div>
  <div className="line"></div>
  <div className="line"></div>
      <div className="Signin-container">
        <h2>Sign In Here</h2>
        <form>
          {/* Username Field */}
          <div className="input-group">
            {/* <label>Username</label> */}
            <input type="text" placeholder="Username" required />
          </div>

          {/* Email Field */}
          <div className="input-group">
            {/* <label>Email</label> */}
            <input type="email" placeholder="Email" required />
          </div>

          {/* Password Field */}
          <div className="input-group password-group">
            {/* <label>Password</label> */}
            <input
              type={passwordVisible ? "text" : "password"}
              placeholder="Password"
              required
            />
            <span className="eyee-icon" onClick={togglePasswordVisibility}>
              {passwordVisible ? <FaEyeSlash /> : <FaEye />}
            </span>
          </div>

          {/* Submit Button */}
          <button type="submit" className="signin-btn">Sign Up</button>
        </form>
        
        {/* Sign-up and forgot password links */}
        <div className="signup-links">
          <a  href="/Login" className="sign-up" onClick={handleLoginClick}>Login</a>
        </div>
        
        {/* Social Sign-In Buttons */}
        <div className="social-login">
          <button className="social-btn google-btn">Google</button>
          <button className="social-btn facebook-btn">Facebook</button>
        </div>
      </div>
    </div>
  );
}

export default Signin;
