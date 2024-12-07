import React from "react";

function InputGroup({  type, placeholder, value, onChange, onBlur }) {
  return (
    <div className="input-group">
      
      <input
        type={type}
        placeholder={placeholder}
        value={value}
        onChange={onChange}
        onBlur={onBlur}
      />
    </div>
  );
}

export default InputGroup;
