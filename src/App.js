import logo from './logo.svg';
import './App.css';
import Header from "./components/Header";
import About from "./components/About.js";
import Contact from "./components/Contact.js";
import Experience from "./components/Experience.js";
import Projects from "./components/Projects.js";
function App() {
  return (
    <div className="App">
        <Header />
        <About />
        <Contact />
        <Experience />
        <Projects />
    </div>
  );
}

export default App;