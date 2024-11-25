import React, { useState } from "react";
import { useNavigate } from "react-router-dom";
import axios from "axios";
import "./Sign-in.css";
import InputGroup from "./Components/inputgroup";
import PasswordInput from "./Components/password";
import LoginLinks from "./Components/Link";
import SocialLogin from "./Components/Social";
import Info from "./Components/info";
import Navbar from "./Components/navbar";

function Login() {
  const navigate = useNavigate();

  // State for form fields
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");

  // State for errors
  const [error, setError] = useState("");

  // Handle form submission
  const handleSubmit = async (e) => {
    e.preventDefault();

    if (!email || !password) {
      setError("Email or password is missing");
      return;
    }

    try {
      const response = await axios.post("http://127.0.0.1:8000/api/login/", {
        email,
        password,
      });

      if (response.status === 200) {
        console.log("Success:", response.data);
        // eslint-disable-next-line no-undef
        localStorage.setItem('username', response.data.username);
        // eslint-disable-next-line no-undef
        // localStorage.setItem('token', response.data.access);
        
        // eslint-disable-next-line no-restricted-globals
        // history.push('/chatbot');  // Redirect to Chatbot UI pa
        alert("Login Successfull")
        navigate("/home-page"); // Redirect to dashboard or desired page
      } else {
        console.error("Login failed:", response.data.message);
        alert("Invalid Credentials")
        // setError(response.data.message || "Login failed. Please try again.");
      }
    } catch (err) {
      console.error("Request faileds:", err.response?.data || err.message);
      // setError(err.response?.data?.detail || "An error occurred. Please try again.");
      alert("Invalid Credentials: Try Again")
    }
  };
  // Handle Signup button click
  const handleSignupClick = () => {
    
  };

  return (
    <div>
      <Navbar />
    <div className="login-page">
      <div className="circle"></div>
      <div className="circle"></div>
      <div className="circle"></div>
      <div className="line"></div>
      <div className="line"></div>
      <div className="line"></div>
      <div className="login-container">
        <h2>Welcome To Architexture</h2>
        <h4>Login Here</h4>
        <form onSubmit={handleSubmit}>
          <InputGroup
            type="email"
            placeholder="Email or Username"
            name="email"
            required
            value={email}
            onChange={(e) => setEmail(e.target.value)}
          />
          <PasswordInput
            type="password"
            name="password"
            placeholder="Password"
            required
            value={password}
            onChange={(e) => setPassword(e.target.value)}
          />
          <button type="submit" className="login-btn">
            Login
          </button>
        </form>

        {error && <p className="error-message">{error}</p>}

        <LoginLinks onSignupClick={handleSignupClick} />
        <SocialLogin />
      </div>
      {/* <div>
        <Info />
      </div> */}
    </div>
    </div>
  );
}

export default Login;
