import {AiFillFilePdf, AiFillGithub, AiFillLinkedin} from "react-icons/ai"

function Header () {

    return (
        <div>
            
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
                    <div className="icon--wrapper resume--icon">
                        <a target="_blank" href="https://drive.google.com/file/d/17Okl_7gg8e3_tMoFIsyFxeONLK1v_IHF/view?usp=sharing">
                            <div className="icon">
                                <AiFillFilePdf size='50px'/> 
                            </div>
                        </a>
                    </div>
                    <div className="icon--wrapper github--icon">
                        <a target="_blank" href="https://github.com/kingsleykimm/">
                            <div className="icon">
                                <AiFillGithub size='50px'/> 
                            </div>
                            
                        </a>
                    </div>
                    <div className="icon--wrapper linkedin--icon">
                        <a target="_blank" href="https://www.linkedin.com/in/kingsleykim/">
                            <div className="icon">
                                <AiFillLinkedin size='50px'/> 
                            </div>  
                        </a>
                    </div>  
                </div>
            </div>
        </div>
        
    )
}

export default Header;