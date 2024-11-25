import React from "react";
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import Login from './Sign-in';
import Signup from './Sign-up';
import Home from './home';

function App() {
  return (
    
    <Router>
      <Routes>
      <Route path="/" element={<Login />} />
        <Route path="/Login" element={<Login />} />
        <Route path="/Sign-in" element={<Signup />} />
        <Route path="/home-page" element={<Home/>}/>
      </Routes>
    </Router>
  );
}
export default App;
