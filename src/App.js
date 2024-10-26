import React from "react";
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import Login from './Login';
import Signup from './Sign-up';

function App() {
  return (
    
    <Router>
      <Routes>
      <Route path="/" element={<Login />} />
        <Route path="/Login" element={<Login />} />
        <Route path="/Sign-in" element={<Signup />} />
      </Routes>
    </Router>
  );
}

export default App;
