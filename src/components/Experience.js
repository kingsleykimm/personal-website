
import { VerticalTimeline, VerticalTimelineElement } from "react-vertical-timeline-component";
import 'react-vertical-timeline-component/style.min.css';
import {MdSchool, MdAnalytics} from "react-icons/md"
import React from "react";
import {Fade} from "react-awesome-reveal"

// Timeline code was cited from https://www.cluemediator.com/how-to-create-a-vertical-timeline-component-in-react
function Experience () {
    const researchIcon = {
        icon: <MdAnalytics />,
        iconStyle: {background: 'rgb(233, 30, 99)', color:'#fff'}
    }
    const schoolIcon = {
        icon : <MdSchool />,
        iconStyle: {background: 'rgb(33, 150, 243)', color: "#fff"}
    }
    const timeline = [
        {
            icon: researchIcon,
            date: "May 2023 - Dec 2023",
            title: 'UVA Biocomplexity Institute AI Research Intern',
            subtitle: 'University of Virginia',
            desc: 'AI, Python, Machine Learning, using machine learning to solve applicable, real-world problems and examining datasets like economic time-series or rainfall'
        },
        {
            icon: schoolIcon,
            date: "Aug 2022 - present",
            title: "University of Virginia",
            subtitle: "Charlotesville, VA",
            desc: "BS in CS + BA in Math"
        },
        {
            icon: researchIcon,
            date: "Jan 2020 - May 2022",
            title: 'Plavchan Group Research Assistant',
            subtitle: 'George Mason University, Fairfax, VA',
            // desc: 'Worked , and used Python to process the images and signals, also used the software AstroImageJ.'
            desc: <ul>
                <li>Worked with the Plavchan research group in analyzing astronomical images to find possible exoplanet candidates, had my own research publication as well</li>
                <li>Used Python for signal processing and noise reduction in images</li>
            </ul>
        },
        {
            icon: schoolIcon,
            date: "Aug 2018 - Jun 2022",
            title: "Thomas Jefferson High School for Science and Technology",
            subtitle: "Alexandria, VA",
            desc: "Top magnet school in the country "
        }
        
    ]
    return (
        <div id="experience">
            <Fade direction='"up"' duration="1500">
                <h1 className="experience--header">
                    Experience & Education
                </h1>
            </Fade>
            <VerticalTimeline>
                {timeline.map((t, i) => {
                const contentStyle = i === 0 ? { background: 'rgb(33, 150, 243)', color: '#fff' } : undefined;
                const arrowStyle = i === 0 ? { borderRight: '7px solid  rgb(33, 150, 243)' } : undefined;
        
                return <VerticalTimelineElement
                    key={i}
                    className="vertical-timeline-element--work"
                    // contentStyle={contentStyle}
                    // contentArrowStyle={arrowStyle}
                    date={t.date}
                    {...t.icon}
                >
                    {t.title ? <React.Fragment>
                    <h3 className="vertical-timeline-element-title">{t.title}</h3>
                    {t.subtitle && <h4 className="vertical-timeline-element-subtitle">{t.subtitle}</h4>}
                    {t.desc && <p>{t.desc}</p>}
                    </React.Fragment> : undefined}
                </VerticalTimelineElement>
                })}
            </VerticalTimeline>
            <p>
                <em>Made with <a href="https://github.com/stephane-monnot/react-vertical-timeline" target="_blank">
                    Vertical Timeline React </a>
                </em>
            </p>
        </div>
    )
}

export default Experience;