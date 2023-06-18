//check commit
import {Fade} from "react-awesome-reveal"
function About () {
  return (
      
      <div id="about">
          <Fade direction='"up"' duration='1500'>
            <h1 className="about--title">
              About Me
            </h1>
          </Fade>
          <Fade direction='"up"' duration='2500'>
            <p className="about--description">I'm a current second-year student at the University of Virginia pursuing a BS in Computer Science, and
              have been coding since my freshman year of high school. Throughout my coding, software development and machine 
              learning have been the two areas that stuck out to me, and where I'm most interested in. 
              I've interned at George Mason University as a research assistant,
              where I used Python to analyze astronomical data, and am currently interning at UVA's "AI For Science" research group, where our group is applying machine
              learning to solve real world problems.
            </p>
          </Fade>
        </div>

    )
};

export default About;