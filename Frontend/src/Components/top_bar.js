// TopBar.jsx
import React from 'react';
import '../chat.css';

const TopBar = () => {
  return (
    <div className="topbar">
      <div className="breadcrumbs">
        Pages / <span>Chat UI</span>
      </div>
      <div className="actions">
        <input type="search" placeholder="Search" className="search-bar" />
        <button className="icon-button">🔒</button>
        <button className="icon-button">🌞</button>
        <div className="profile-icon">AP</div>
      </div>
    </div>
  );
};

export default TopBar;
