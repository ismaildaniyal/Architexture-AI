import React from "react";
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import Signup from './pages/Sign-up';
import Home from './pages/home';
import Login from './pages/Sign-in'
// import Chat from './chat'
function App() {
  return (
    
    <Router>
      <Routes>
      <Route path="/" element={<Login/>} />
        <Route path="/Login" element={<Login />} />
        <Route path="/Sign-in" element={<Signup />} />
        <Route path="/home-page" element={<Home/>}/>
      </Routes>
    </Router>
  );
}
export default App;
