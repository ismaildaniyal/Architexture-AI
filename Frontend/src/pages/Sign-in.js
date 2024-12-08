// import 'bootstrap/dist/css/bootstrap.min.css';
import React, { useState, useEffect } from "react";
import { Link, useNavigate } from "react-router-dom";
import axios from "axios";
import "../styles/Sign-in.css";
import InputGroup from "../Components/inputgroup";
import PasswordInput from "../Components/password";
import LoginLinks from "../Components/Link";
// import SocialLogin from "./Components/Social";
import Navbar from "../Components/navbar";
// import PasswordStrengthBar from 'react-password-strength-bar';
import Cookies from 'js-cookie'; // Import js-cookie

// import Info from './Components/info'

function Login() {
  const navigate = useNavigate();
  const [otp, setOtp] = useState(new Array(4).fill("")); // State for OTP inputArra
  //const otpRefs = useRef([]);

  // State for form fields
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [isForgotPassword, setIsForgotPassword] = useState(false); // Toggle state for forgot password form
  const [otpView, setOtpView] = useState(false); // Toggle state for OTP input view
  // const [error, setError] = useState("");

  const [isResetPassword, setIsResetPassword] = useState(false); // Toggle state for reset password form
  const [newPassword, setNewPassword] = useState(""); // State for new password
  const [confirmPassword, setConfirmPassword] = useState(""); // State for confirm password
  const [isLoading, setIsLoading] = useState(false);
  
  // Timer states
  const [minutes, setMinutes] = useState(0);
  const [seconds, setSeconds] = useState(60);
  const [isTimerActive, setIsTimerActive] = useState(false);
  const [isOtpExpired, setIsOtpExpired] = useState(false);


  // Timer logic'
  useEffect(() => {
    let timer;
    if (isTimerActive) {
      timer = setInterval(() => {
      if (seconds > 0) {
        setSeconds(seconds - 1);
      } else if (minutes > 0) {
        setMinutes(minutes - 1);
        setSeconds(59);
      } else {
        clearInterval(timer);
        setIsOtpExpired(true);
        setOtpView(false);
        // alert("Time expired. Please request a new OTP.");
        setIsForgotPassword(true);
      }
      }, 1000);
    }

    return () => clearInterval(timer);
  }, [isTimerActive, minutes, seconds]);

  useEffect(() => {
    if (otpView) {
      setIsTimerActive(true);
      setMinutes(0);
      setSeconds(60);
      setIsOtpExpired(false);
    } else {
      setIsTimerActive(false);
      setMinutes(0);
      setSeconds(60);
    }
  }, [otpView]);


  
           
  
  
  

const form_change =()=>{
  setEmail("")
  setIsForgotPassword(true)

}
const password_form= ()=>{
  setEmail("")
  setIsForgotPassword(false)
}
  // Handle OTP input change
  const handleOtpChange = (e, index) => {
    const value = e.target.value; // Get the value from the event object
    const newOtp = [...otp]; // Create a copy of the otp array
    newOtp[index] = value; // Update the value at the specified index
    setOtp(newOtp); // Update the OTP state with the new value

    if(value && e.target.nextSibling)
    {
      e.target.nextElementSibling.focus();
    }
  };
  const handleKeyDown = (e, index) => {
    // Handle arrow keys for navigation
    if (e.key === 'ArrowRight')
   {
      // Move focus to the next sibling if available
      if (e.target.nextElementSibling) 
      {
        e.target.nextElementSibling.focus();
      }
    } else if (e.key === 'ArrowLeft') 
    {
      // Move focus to the previous sibling if available
      if (e.target.previousElementSibling)
      {
        e.target.previousElementSibling.focus();
      }
    }
    if (e.target.value && e.target.setSelectionRange) 
    {
      e.target.setSelectionRange(1, 1);  // Set cursor to the start (before the digit)
    }
  };
  

  
  // Handle form submission Login form
  const handleSubmit = async (e) => {
    e.preventDefault();
   
    if (!email || !password) {
      alert("Please enter both email and password");
      return;
    }

    try {
      const response = await axios.post("http://127.0.0.1:8000/api/login/", {
        email,
        password,
      });

      if (response.status === 200) {
        // Set the username in cookies
        Cookies.set('username', response.data.username, { secure: true, sameSite: 'Strict' });
        Cookies.set('user_session', response.data.token, { secure: true, sameSite: 'Strict' });

        alert("Login Successful");
        navigate("/home-page");
      } else {
        // console.error("Login failed:", response.data.message);
        alert("Invalid Credentials");
      }
    } catch (err) {
      alert("Invalid Credentials: Try Again");
    }
  };

  // Handle Forgot Password form submission 
  const handleForgotPasswordSubmit = async (e) => {
    e.preventDefault();
  
    if (!email) {
      // setError("Please enter your email.");
      alert("Please enter your email")
      return;
    }
    setIsLoading(true); // Show spinner
    try {
      const response = await axios.post(
        "http://127.0.0.1:8000/api/generate-otp/",
        { email },
        {
          headers: {
            "Content-Type": "application/json",
          },
        }
      );
      if (response.status === 200) {
        setOtpView(true); // Show OTP input form
        // alert(OTP has been sent to your email.);
        Cookies.set("email", email, { secure: true, sameSite: "Strict" });
        setOtp(["", "", "", ""]); // Clear OTP input fields
      } else {
        alert("Failed to send OTP. Please try again.")
        // setError("Failed to send OTP. Please try again.");
      }
      
    } catch (err) {
      if (err.response) {
        // Server responded with a status other than 200 range
        const status = err.response.status;
        if (status === 404) {
          // setError(err.response.data.error); // "Account does not exist. Please sign up."
          alert("Account does not exist. Please sign up.")
          navigate("/Sign-in")
        } else if (status === 400) {
          alert("Account does not exist. Please sign up.")
          navigate("/Sign-in")
        } else if (status === 500) {
          alert("An error occurred while sending OTP. Please try again.")
          // setError("An error occurred while sending OTP. Please try again.");
        } else {
          alert("Unexpected error occurred. Please try again.")
          // setError("Unexpected error occurred. Please try again.");
        }
      } else {
        // No response from server
        alert("Unable to connect to the server. Please check your network connection.")
        // setError("Unable to connect to the server. Please check your network connection.");
      }
    }finally {

      setIsLoading(false); // Hide spinner
    }
  };
  

  // Handle OTP submission
  const handleOtpSubmit = async (e) => {
    e.preventDefault();
  
    const enteredOtp = otp.join(""); // Join OTP array to form the full OTP string
    const email = Cookies.get("email"); // Get email from cookies (assuming it was stored previously)
  
    if (!enteredOtp || !email) {
      alert("Please enter the OTP and ensure email is present.")
      // setError("Please enter the OTP and ensure email is present.");
      return;
    }
  
    try {
      const response = await axios.post(
        "http://127.0.0.1:8000/api/verify-otp/",
        { email, otp: enteredOtp },  // Ensure correct fields are sent
        {
          headers: {
            "Content-Type": "application/json",
          },
        }
      );
  
      if (response.status === 200) {
        alert("OTP Verified! Proceeding to reset password.");
        setIsResetPassword(true); // Show reset password form
      }
      
    } catch (err) {
      if (err.response) {
        const { status, data } = err.response;
  
        if (status === 408) {
          // alert("OTP has expired. Please re-enter your email to request a new one.");
          Cookies.remove("email"); // Clear the email from cookies
          setOtpView(false); // Hide OTP input view
          setIsForgotPassword(true); // Show forgot password form
        } else if (status === 406) {
          alert("Invalid otp")
         // setError("Invalid OTP. Please try again."); // Show error, keep form visible
          setOtp(["", "", "", ""]); // Optionally, clear OTP fields
        } else {
          alert(data.error || "An unexpected error occurred. Please try again.")
          // setError(data.error || "An unexpected error occurred. Please try again.");
        }
      } else {
        // console.error("Error:", err.message);
        alert("An unexpected error occurred. Please check your network connection and try again.")
        // setError("An unexpected error occurred. Please check your network connection and try again.");
      }
    }
  };
  
  const validatePassword1 = (value) => {
    const passwordRegex =/^(?=.*?[A-Z])(?=.*?[a-z])(?=.*?[0-9])(?=.*?[#?!@$%^&*-]).{8,}$/;
  return passwordRegex.test(value);
  };
  // Handle Reset Password form submission
  const handleResetPasswordSubmit = async (e) => {
    e.preventDefault();
  
    if (!newPassword || !confirmPassword) {
      alert("Please fill in all fields.")
      // setError("Please fill in all fields.");
      return;
    }
    if (newPassword !== confirmPassword) {
      alert("Passwords do not match.")
      // setError("Passwords do not match.");
      return;
    }
    if (!validatePassword1(newPassword)) {
      alert("Invalid password format. Please have must have Capitial,Small,special,Number.");
      return;
    }
  
    try {
      const response = await axios.post("http://127.0.0.1:8000/api/update-password/", {
        email,
        password: newPassword, // Match backend expectation
      });
  
      if (response.status === 200) {
        alert("Password reset successful! Please log in with your new password.");
        setConfirmPassword("")
        setNewPassword("")
        setEmail("")
        setPassword("")
        setIsForgotPassword(false);
        setOtpView(false);
        setIsResetPassword(false);
        
      } else {
        setConfirmPassword("")
        setNewPassword("")
        
        alert("Failed to reset password. Please try again.")
        // setError("Failed to reset password. Please try again.");
      }
    } catch (err) {
      setConfirmPassword("")
        setNewPassword("")
      alert("An error occurred. Please try again.")
      // setError("An error occurred. Please try again.");
    }
  };
 
  useEffect(() => {
    // Clear all session cookies on component mount
    Cookies.remove("username");
    Cookies.remove("user_session");
    Cookies.remove("email");
  }, []);
  
 
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
          {!isForgotPassword ? (
            <>
              <h2>Welcome To Architexture</h2>
              <h4>Login Here</h4>
              {/* Login form submit */}
              <form onSubmit={handleSubmit}>
                
                <InputGroup
                  type="email"
                  placeholder="Email"
                  name="email"
                  value={email}
                  onChange={(e) => setEmail(e.target.value)}
                  
                />
                <div className="error-email">
                {/* {error_email && <p className="error-message-email">{error_email}</p>} */}
                </div>
                <PasswordInput
                  type="password"
                  name="password"
                  placeholder="Password"
                  required
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                />
                {/* <PasswordStrengthBar password={password} /> */}
                  <LoginLinks
                onSignupClick={() => {}}
                onForgotPasswordClick={()=>form_change()} // Show forgot password form
              />
                <button type="submit" className="login-btn">
                  Login
                </button>
                <div className="Sign-up-links">
               <p>Do not have an account? <Link to="/Sign-in">Sign Up</Link></p>
          </div>
              </form>
            
              
            </>
          ) : isResetPassword ? (
            <>
              <h2>Reset Password</h2>
              <h4>Enter your new password</h4>
              
              <form onSubmit={handleResetPasswordSubmit}>
                <PasswordInput
                  type="password"
                  name="newPassword"
                  placeholder="New Password"
                  required
                  value={newPassword}
                  onChange={(e) => setNewPassword(e.target.value)}
                />
                
                {/* <PasswordStrengthBar password={password} /> */}
                <PasswordInput
                  type="password"
                  name="confirmPassword"
                  placeholder="Confirm Password"
                  required
                  value={confirmPassword}
                  onChange={(e) => setConfirmPassword(e.target.value)}
               
                />
                
                {/* <PasswordStrengthBar password={password} /> */}
                <button type="submit" className="login-btn">
                  Reset Password
                </button>
              </form>
              
            </>
          ) : otpView ? (
            <>
              <h2>Enter OTP</h2>
              <h4>We have sent an OTP to your email</h4>
              <form onSubmit={handleOtpSubmit}>
              <div className="otp-instructions">
                  
                  <p style={{ color: "white", marginTop: "-10px" }}>
                        {isOtpExpired
                          ? "OTP expired. Please request a new one."
                          : `Time remaining: ${minutes}:${seconds < 10 ? `0${seconds}` : seconds}`}
                      </p>
                </div>
                <div className="otp-inputs">
                  {otp.map((value, index) => (
                    <input
                      //key={index}
                      type="text"
                      maxLength="1"
                      className="otp-input"
                      value={value}
                      onChange={(e) => handleOtpChange(e, index)}
                      onKeyDown={(e) => handleKeyDown(e, index)} // Handle arrow keys
                    />
                  ))}
                </div>
                  
                <button type="submit" className="login-btn"  disabled={isOtpExpired}>
                  Verify OTP
                </button>
              </form>
              
            </>
          ) : (
            <>
              <h2>Forgot Password</h2>
              <h4>Enter your email to reset your password</h4>
              <form onSubmit={handleForgotPasswordSubmit}  className={isLoading ? "blurred" : ""}>
                <InputGroup
                  type="email"
                  placeholder="Enter your email"
                  name="email"
                  required
                  value={email}
                  onChange={(e) => setEmail(e.target.value)}
                />
                <div className="Link">
               <Link
                  onClick={() => password_form()}
                  className="back-to-login-link">
                    
                  Back to Login
                </Link>
                </div>
                <button type="submit" className="login-btn" >
                    "Submit"
                  
                </button>
              </form>
              {isLoading && <div className="spinner-overlay">
                <div className="spinner"></div>
                </div>}
             

              <div>
          
              </div>
            </>
          )}
        </div>
      </div>
    </div>
  );
}

export default Login;
