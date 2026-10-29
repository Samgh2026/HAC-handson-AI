# =============================================================================
# FILE: crewai_handoff_demo.py
# PURPOSE : Show the simplest agent-to-agent handoff in CrewAI.
# PROBLEM : Agent 1 (researcher) reads a website; Agent 2 (writer) must
#           write a blog post using ONLY what Agent 1 found.
# HOW     : Task 2 declares context=[research_task]. When Task 1 finishes,
#           CrewAI injects its output into Task 2's prompt -> that is the handoff.
# RUN     : pip install crewai crewai-tools
#           export OPENAI_API_KEY="sk-..."   (used for LLM + embeddings)
#           python crewai_handoff_demo.py
# OUTPUT  : Researcher notes printed during the run, then a markdown blog post.
# =============================================================================

from crewai import Agent, Task, Crew, Process   # core CrewAI building blocks
from crewai_tools import WebsiteSearchTool       # RAG search over one website

TARGET_URL = "https://docs.crewai.com/"          # website the researcher will read

# Tool: embeds the site's pages, then lets the agent semantically search them
website_tool = WebsiteSearchTool(website=TARGET_URL)

# -------------------- Agent 1: Website Researcher ----------------------------
researcher = Agent(
    role="Website Researcher",
    goal=f"Extract key facts from {TARGET_URL}",
    backstory="You read websites carefully and report only what is on the page.",
    tools=[website_tool],          # only the researcher can touch the website
    allow_delegation=False,        # no free-form delegation; handoff is explicit
    verbose=True,
)

# -------------------- Agent 2: Blog Writer -----------------------------------
writer = Agent(
    role="Blog Writer",
    goal="Turn research notes into a clear, beginner-friendly blog post",
    backstory="You write simple blog posts and never invent facts.",
    tools=[],                      # no tools: writer relies purely on the handoff
    allow_delegation=False,
    verbose=True,
)

# -------------------- Task 1: Research ---------------------------------------
research_task = Task(
    description=(
        f"Search {TARGET_URL} and collect the 5 most important facts: "
        "what it is, main features, and who it is for."
    ),
    expected_output="A bullet list of 5 facts, each one line.",
    agent=researcher,
)

# -------------------- Task 2: Write (receives the handoff) -------------------
writing_task = Task(
    description=(
        "Using ONLY the research notes provided in your context, "
        "write a 300-word blog post with a title and 3 short sections."
    ),
    expected_output="A markdown blog post.",
    agent=writer,
    context=[research_task],       # <-- THE HANDOFF: researcher output -> writer
)

# -------------------- Crew: wires agents + tasks together --------------------
crew = Crew(
    agents=[researcher, writer],
    tasks=[research_task, writing_task],   # run order: research first, then write
    process=Process.sequential,            # one task after another
    verbose=True,
)

if __name__ == "__main__":
    result = crew.kickoff()                # starts the pipeline
    print("\n===== FINAL BLOG POST =====\n")
    print(result.raw)                      # .raw = final task's text output
