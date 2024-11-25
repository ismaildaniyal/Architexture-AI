import React, { useState } from "react";
import { useNavigate } from "react-router-dom"; // Import useNavigate from react-router-dom
import InputGroup from "./Components/inputgroup";
import PasswordGroup from "./Components/password";
import SocialButtons from "./Components/Social";
import Info from "./Components/info";
import axios from "axios";
import "./Sign-up.css";
import Navbar from "./Components/navbar";

// Axios Configuration
axios.defaults.xsrfCookieName = "csrftoken";
axios.defaults.xsrfHeaderName = "X-CSRFToken";
axios.defaults.withCredentials = true;

function Signin() {
  // const [currentuser, setcurrentuser] = useState(null); // State to track logged-in user
  const [username, setUsername] = useState(""); // State for username
  const [email, setEmail] = useState(""); // State for email
  const [password, setPassword] = useState(""); // State for password

  const navigate = useNavigate(); // Initialize useNavigate hook within the function

  // Handle form submission
  const handleSubmit = async (e) => {
    e.preventDefault(); // Prevent page reload

    // Debugging: Log the values of the fields to verify they are populated
    // console.log("Username:", username);
    // console.log("Email:", email);
    // console.log("Password:", password);

    // Simple client-side validation to ensure no field is empty
    if (!username || !email || !password) {
      console.error("All fields must be filled!");
      return; // Stop the submission if any field is empty
    }

    try {
      const response = await axios.post("http://127.0.0.1:8000/api/signup/", {
        username,
        email,
        password,
      });
      console.log("User signed up successfully:", response.data);
      alert("User Created Successful")
      // setcurrentuser(response.data); // Set current user after successful signup

      // Navigate to Sign-in page upon successful signup
      navigate("/Login"); // Redirect to Sign-in page

    } catch (error) {
      if (error.response) {
        console.error("Error signing up:", error.response.data); // Server response
      } else {
        console.error("Error signing up:", error.message); // Network or other errors
      }
    }
  };

  return (
    <div>
      <Navbar />
    <div className="Signin-page">
      {/* <div>
        <Info />
      </div> */}
      <div className="circle"></div>
      <div className="circle"></div>
      <div className="circle"></div>
      <div className="line"></div>
      <div className="line"></div>
      <div className="line"></div>
      <div className="Signin-container">
        <h2>Sign Up Here</h2>
        <form onSubmit={handleSubmit}>
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
          <button type="submit" className="signin-btn">
            Sign Up
          </button>
        </form>
        <div className="signup-links"></div>
        <SocialButtons />
      </div>
    </div>
    </div>
  );
}

export default Signin;
