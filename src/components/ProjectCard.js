

export default function ProjectCard (props) {
    return (
        <div className="card--container">
            <img src= {props.image} alt="Project Image"></img>
            <div className="card-description-container">
                <p className="card--description">{props.desc}</p>
                <p>Skills: {props.skills}</p>
            </div>
            <div className="card--links">
                {props.siteLink && <a className="card-link" href={props.siteLink}>Site Link</a>}
                {props.gitLink && <a className="card-link" href={props.gitLink}>Github</a>}
            </div>
        </div>        
    )
}