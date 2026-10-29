'''
blog writer agent 
1 agent --> 1 task 
'''

import os
from dotenv import load_dotenv

load_dotenv(override=True)   
from crewai import Agent, Task, Crew

from crewai_tools import WebsiteSearchTool

# create the tool object
search_tool = WebsiteSearchTool()

# define the agent 
writer_agent = Agent(
    role = "Blog writer agent",
    goal = "Write short clear blog posts, for social media based on latest research",
    backstory = "You can explain complex concepts with ease using simple analogies",
    llm = "gpt-4o-mini",
    tools = [search_tool] # agent uses search tool to perform the task
    
)

# define the task : 
write_task = Task(
    description = "Write a 350 word blog post about the {topic}",
    expected_output = "A blog post for Linkedin, with clickable title, 4 paragraphs",
    agent = writer_agent # assigning this task to writer agent
)

# assemble the crew
crew = Crew(
    agents = [writer_agent],
    tasks = [write_task],
    verbose = True
)

# result 
result = crew.kickoff(inputs = {"topic" : "Latest NVIDIA news as of September 2026"})

print(result)