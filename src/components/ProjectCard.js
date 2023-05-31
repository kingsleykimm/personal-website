

export default function ProjectCard (props) {
    return (
        <div className="card--container">
            <h2 style={{color: "#5d5e60"}}>{props.title}</h2>
            <img src= {props.image} alt="Project Image"></img>

            <div className="card-description-container">
                <p className="card--description">{props.desc}</p>
            </div>
            <div className="card-skills-container">
            {
                props.skills.map(item => {
                    return (
                    
                        <div className="card--skills">{item}</div>
                    
                    )
                })
            }
            </div>
            <div className="card--links">
                {props.siteLink && <a className="card-link" href={props.siteLink}>Site Link</a>}
                {props.gitLink && <a className="card-link" href={props.gitLink}>Github</a>}
            </div>
        </div>        
    )
}