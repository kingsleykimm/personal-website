
import { VerticalTimeline, VerticalTimelineElement } from "react-vertical-timeline-component";
import 'react-vertical-timeline-component/style.min.css';
import {MdSchool, MdAnalytics} from "react-icons/md"
import React from "react";

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
            title: '\"AI For Science\" Research Intern',
            subtitle: 'University of Virginia',
            desc: 'AI, Python, Machine Learning'
        },
        {
            icon: schoolIcon,
            date: "Aug 2022 - present",
            title: "University of Virginia",
            subtitle: "Charlotesville, VA"
        },
        {
            icon: researchIcon,
            date: "Jan 2020 - May 2022",
            title: 'Plavchan Group Research Assistant',
            subtitle: 'George Mason University, Fairfax, VA',
            desc: 'Research, Python, Data Analysis, Signal Processing'
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
            <h1 className="experience--header">
                Experience & Education
            </h1>
            <h2 className="experience-header">
                The places I've worked and studied at
            </h2>
            <VerticalTimeline>
                {timeline.map((t, i) => {
                const contentStyle = i === 0 ? { background: 'rgb(33, 150, 243)', color: '#fff' } : undefined;
                const arrowStyle = i === 0 ? { borderRight: '7px solid  rgb(33, 150, 243)' } : undefined;
        
                return <VerticalTimelineElement
                    key={i}
                    className="vertical-timeline-element--work"
                    contentStyle={contentStyle}
                    contentArrowStyle={arrowStyle}
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
        </div>
    )
}

export default Experience;