from imaplib import Commands
import os
from dotenv import load_dotenv
from openai import BaseModel
from pydantic import Field, validator
import requests
from sqlalchemy import literal
load_dotenv()
from typing import Literal
from langchain_openai import AzureChatOpenAI
from langgraph.graph import StateGraph, START, END, MessagesState
from langchain_core import HumanMessage, SystemMessage
from langgraph.types import Command
from langgraph.prebuilt import create_react_agent
from langchain.tools import tool




llm = AzureChatOpenAI(
    openai_api_key=os.environ["AZURE_OPENAI_API_KEY"],
    openai_api_version=os.environ["AZURE_OPENAI_API_VERSION"],
    azure_deployment=os.environ["AZURE_OPENAI_CHAT_DEPLOYMENT_NAME"],
    azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
    openai_api_type="azure",
    temperature=1,
)


class Supervisor(BaseModel):
    next: Literal["enhancer", "researcher", "coder"] = Field(
        description="Determines which specialist to activate next in the workflow sequence: "
                    "'enhancer' when user input requires clarification, expansion, or refinement, "
                    "'researcher' when additional facts, context, or data collection is necessary, "
                    "'coder' when implementation, computation, or technical problem-solving is required."
    )
    reason: str = Field(
        description="Detailed justification for the routing decision, explaining the rationale behind selecting the particular specialist and how this advances the task toward completion."
    )

def supervisor_node(state: MessagesState) ->Command[Literal["enhancer", "researcher", "code"]]:
    system_prompt = (
        """
        You are a workflow supervisor managing a team of three specialized agents: Prompt Enhancer, Researcher, and Coder. Your role is to orchestrate the workflow by selecting the most appropriate next agent based on the current state and needs of the task. Provide a clear, concise rationale for each decision to ensure transparency in your decision-making process.

        **Team Members**:
        1. **Prompt Enhancer**: Always consider this agent first. They clarify ambiguous requests, improve poorly defined queries, and ensure the task is well-structured before deeper processing begins.
        2. **Researcher**: Specializes in information gathering, fact-finding, and collecting relevant data needed to address the user's request.
        3. **Coder**: Focuses on technical implementation, calculations, data analysis, algorithm development, and coding solutions.

        **Your Responsibilities**:
        1. Analyze each user request and agent response for completeness, accuracy, and relevance.
        2. Route the task to the most appropriate agent at each decision point.
        3. Maintain workflow momentum by avoiding redundant agent assignments.
        4. Continue the process until the user's request is fully and satisfactorily resolved.

        Your objective is to create an efficient workflow that leverages each agent's strengths while minimizing unnecessary steps, ultimately delivering complete and accurate solutions to user requests.

"""
    )
    messages = [{
        "role":"system","content": system_prompt
    }]+ state["messages"]

    structured_output = llm.with_structured_output(Supervisor)
    response = structured_output.invoke(messages)

    goto = response.next
    reason = response.reason
    return Command(
        update={
            "messages":[
                HumanMessage(content=reason, name="supervisor")
            ]
        },
        goto = goto
    )


def enhancer(state: MessagesState) ->Command[Literal["supervisor_node"]]:
    """
        Enhancer agent node that improves and clarifies user queries.
        Takes the original user input and transforms it into a more precise,
        actionable request before passing it to the supervisor.
    """
    system_prompt = (

        "You are a Query Refinement Specialist with expertise in transforming vague requests into precise instructions. Your responsibilities include:\n\n"
        "1. Analyzing the original query to identify key intent and requirements\n"
        "2. Resolving any ambiguities without requesting additional user input\n"
        "3. Expanding underdeveloped aspects of the query with reasonable assumptions\n"
        "4. Restructuring the query for clarity and actionability\n"
        "5. Ensuring all technical terminology is properly defined in context\n\n"
        "Important: Never ask questions back to the user. Instead, make informed assumptions and create the most comprehensive version of their request possible."
    )
    messages = [
        {"role": "system", "content": system_prompt}
        ] + state["messages"]
    enhanced_query = llm.invoke(messages)
    return Command(
        update= {
            "messages":[
                HumanMessage(content=enhanced_query.content, name="enhancer")
            ]
        },
        goto = "supervisor_node"
    )
@tool("tavily_search", return_direct=True)
def tavily_search(query: str):

    url = "https://api.tavily.com/search"
    headers = {
        "Authorization": "YOUR_TAVILY_API_KEY"
    }
    params = {
        "query": query,
        "num_results": 3
    }
    response = requests.post(url, headers = headers, params=params)
    data = response.json()
    results = data.get("results",[])
    return {
        "status": "successfull",
        "results": results

    }

def researcher_node(state: MessagesState) -> Command[Literal["validator"]]:
    """
    Research agent node that gathers information using Tavily search.
    Takes the current task state, performs relevant research,
    and returns findings for validation.
    """

    research_agent = create_react_agent(
        llm,
        tools=["tavily_search"],
        prompt=(
            "You are an Information Specialist with expertise in comprehensive research. "
            "Your responsibilities include:\n"
            "1. Identifying key information needs based on the query context\n"
            "2. Gathering relevant, accurate, and up-to-date information from reliable sources\n"
            "3. Organizing findings in a structured, easily digestible format\n"
            "4. Citing sources when possible to establish credibility\n"
            "5. Focusing exclusively on information gathering - avoid analysis or implementation\n\n"
            "Provide thorough, factual responses without speculation where information is unavailable."
        )
    )

    result = research_agent.invoke(state)
    return Command(
        update = {
            "messages" : [
                HumanMessage(content= result["messages"][-1].content, name= "researcher")
            ]
        },
        goto = "validator"
    )

@tool("python_repl_tool", return_direct=True)
def python_repl_tool(code: str) -> Dict[str, Any]:
    """
    Executes the given Python code string in a REPL-like environment.
    Useful for math, data manipulation, or general Python scripting.
    Returns the result or error.
    """
    try:
        # Create a safe namespace for execution
        local_vars = {}
        exec(code, {}, local_vars)
        return {"output": str(local_vars)}
    except Exception as e:
        return {"error": str(e)}

def code_node(state: MessagesState) -> Command[literal["validator"]]:
    code_agent = create_react_agent(
        llm,
        tools= ["python_repl_tool"],
        prompt = "You are a coder and analyst. Focus on mathematical calculations, analyzing, solving math questions, "
            "and executing code. Handle technical problem-solving and data tasks."
    )
    response = code_agent.invoke(state)
    return Command(
        update = {
            "messages": [
                HumanMessage(content= response["messages"][-1].content, name= "coder" )
            ]
        },
        goto = validator
    )

