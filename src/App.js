
import './App.css';
import Header from "./components/Header";
import About from "./components/About.js";
import Contact from "./components/Contact.js";
import Experience from "./components/Experience.js";
import Projects from "./components/Projects.js";
import {BrowserRouter} from "react-router-dom";
import {HashLink as Link } from "react-router-hash-link";

function App() {
  return (
    <div className="App">
        <BrowserRouter>
            <header className="page--header">
                <Link className="page--link" smooth to="#about" >Bio</Link>
                <Link className = "page--link" smooth to="#experience">Experience</Link>
                <Link className="page--link" smooth to="#projects">Projects</Link>
                <Link className="page--link" smooth to="#contact">Contact</Link>
            </header>
        </BrowserRouter>
        <Header />
        <About />
        <Experience />
        <Projects />
        <Contact />
        
        <footer style = {{marginBottom: '40px', color: '#5d5e60', fontSize: '16px'}}>
          © 2023 Kingsley Kim
        </footer>
    </div>
  );
}

export default App;
