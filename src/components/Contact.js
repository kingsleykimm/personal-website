
import {AiFillFilePdf, AiFillGithub, AiFillLinkedin, AiOutlineMail} from "react-icons/ai"
function Contact () {
    return (
        <div id="contact">
            <h1 className="contact--header">
                Socials/Contact Me
            </h1>
            <div className="contact--divs">
                <a target="_blank" href="mailto:kingsleykimm@gmail.com" className="contact--div">
                    <AiOutlineMail size='30px'/>
                    <p>Email</p>
                </a>
                <a target="_blank" href="https://github.com/kingsleykimm" className="contact--div">
                    <AiFillGithub size='30px'/>
                    <p>Github</p>
                </a>
                <a target="_blank" href="https://www.linkedin.com/in/kingsleykim/" className="contact--div">
                    <AiFillLinkedin size='30px'/>
                    <p>Linkedin</p>
                </a>
                <a target="_blank" href="https://drive.google.com/file/d/17Okl_7gg8e3_tMoFIsyFxeONLK1v_IHF/view?usp=sharing" className="contact--div">
                    <AiFillFilePdf size='30px'/>
                    <p>Resume</p>
                </a>

         
            </div>
           
        </div>
    )
}
export default Contact;