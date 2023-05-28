
import {BrowserRouter} from "react-router-dom";
import {HashLink as Link } from "react-router-hash-link";
import {AiFillFilePdf, AiFillGithub, AiFillLinkedin} from "react-icons/ai"

function Header () {

    return (
        <div>
            <BrowserRouter>
                <header className="page--header">
                    <Link className="page--link" smooth to="About" >Bio</Link>
                    <Link className = "page--link" smooth to="Experience">Experience</Link>
                    <Link className="page--link" smooth to="Projects">Projects</Link>
                    <Link className="page--link" smooth to="Contact">Contact</Link>
                </header>
            </BrowserRouter>
            <div className="header--wrapper">
                <div className="header--body">
                    <div className="title">
                        <h1><span className="my-name">Kingsley Kim</span>
                        <hr></hr>
                        </h1>
                    </div>
                    <p className="description"><em>Computer Science Student @ UVA passionate about software development & machine learning</em></p>
                </div>
                <div className="social--links">
                    <div className="icon--wrapper">
                        <a target="_blank" href="https://drive.google.com/file/d/17Okl_7gg8e3_tMoFIsyFxeONLK1v_IHF/view?usp=sharing">
                            <div className="icon">
                                <AiFillFilePdf size='40px'/> 
                            </div>
                        </a>
                    </div>
                    <div className="icon--wrapper">
                        <a target="_blank" href="https://github.com/kingsleykimm/">
                            <div className="icon">
                                <AiFillGithub size='40px'/> 
                            </div>
                            
                        </a>
                    </div>
                    <div className="icon--wrapper">
                        <a target="_blank" href="https://www.linkedin.com/in/kingsleykim/">
                            <div className="icon">
                                <AiFillLinkedin size='40px'/> 
                            </div>  
                        </a>
                    </div>  
                </div>
            </div>
        </div>
        
    )
}

export default Header;