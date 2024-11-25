import React from "react";

function InputGroup({  type, placeholder, value, onChange }) {
  return (
    <div className="input-group">
      
      <input
        type={type}
        placeholder={placeholder}
        value={value}
        onChange={onChange}
        required
      />
    </div>
  );
}

export default InputGroup;
