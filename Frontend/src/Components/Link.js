import React from "react";

function LoginLinks({ onSignupClick }) {
  return (
    <div className="login-links">
      <a href="/#" className="forgot-password">Forgot Password?</a>
      <a href="/Sign-in" className="sign-up" onClick={onSignupClick}>Sign Up</a>
    </div>
  );
}

export default LoginLinks;
