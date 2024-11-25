import React, { useState } from "react";
import { FaEye, FaEyeSlash } from "react-icons/fa";

function PasswordInput({ placeholder, value, onChange }) {
  const [passwordVisible, setPasswordVisible] = useState(false);

  const togglePasswordVisibility = () => {
    setPasswordVisible(!passwordVisible);
  };

  return (
    <div className="input-group password-group">
      <input  
        type={passwordVisible ? "text" : "password"}
        placeholder={placeholder}
        required
        value={value} // Pass value prop here
        onChange={onChange} // Pass onChange prop here
      />
      <span className="eye-icon" onClick={togglePasswordVisibility}>
        {passwordVisible ? <FaEyeSlash /> : <FaEye />}
      </span>
    </div>
  );
}

export default PasswordInput;
