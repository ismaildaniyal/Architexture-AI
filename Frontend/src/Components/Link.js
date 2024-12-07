import React from "react";
import { Link } from 'react-router-dom';
function LoginLinks({ onSignupClick, onForgotPasswordClick }) {
  return (
    <div className="login-links">
      <Link href="/#" className="forgot-password" onClick={onForgotPasswordClick}>
        Forgot Password?
      </Link>
      {/* <Link to="/Sign-in" className="sign-up" onClick={onSignupClick}>Sign Up</Link> */}
    </div>
  );
}

export default LoginLinks;
