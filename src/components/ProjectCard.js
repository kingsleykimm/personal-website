

export default function ProjectCard (props) {
    return (
        <div className="card--container">
            <img src= {props.image} alt="Project Image"></img>
            <div className="card--description">
                <p className="card--description">{props.desc}</p>
                <p>{props.skills}</p>
            </div>
            <div className="card--links">
                {props.siteLink && <a href={props.siteLink}>Site Link</a>}
                {props.gitLink && <a href={props.gitLink}>Github</a>}
            </div>
        </div>        
    )
}