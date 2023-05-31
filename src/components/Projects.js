// plan for projects section: make a 2 x n grid for projects, and make individual project cards for each
//take the data from the projectsdata.js file, which is an array of JSON objects, and pass the object's values as props to ProjectCard.js
//Use ProjectCard.js as the
import {Fade} from "react-awesome-reveal"
import projectdata from "./projectdata";
import ProjectCard from "./ProjectCard";

function Projects () {
    const data = projectdata;
    return (
        <div id="projects">
            <Fade direction="up" duration="1500">
                <h1 className="project--header">
                    Projects
                </h1>
            </Fade>
            <Fade direction="up">
                <p className="project--subheader">
                    <em>Some stuff I've worked/been working on</em>
                </p>
            </Fade>
            {
                data.map((item, index) => {

                    return <ProjectCard 
                        image = {item.image}
                        desc = {item.desc}
                        siteLink = {item.siteLink}
                        gitLink = {item.gitLink}
                    />
                })
            }

        </div>
    )
}

export default Projects;