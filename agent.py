import os
from dotenv import load_dotenv
from openai import BaseModel
from pydantic import Field
load_dotenv()

from typing import List, TypedDict
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain.schema import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import AzureChatOpenAI

llm = AzureChatOpenAI(
    openai_api_key=os.environ["AZURE_OPENAI_API_KEY"],
    openai_api_version=os.environ["AZURE_OPENAI_API_VERSION"],
    azure_deployment=os.environ["AZURE_OPENAI_CHAT_DEPLOYMENT_NAME"],
    azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
    openai_api_type="azure",
    temperature=1,
)



class Agentstate(TypedDict):
    messages: List[BaseMessage]
    documents: list[Document]
    proceed_to_generate: bool
    question: str
    rephrased_question: str
    on_topic: str
    rephase_count: str
    
class GradeQuestion(BaseModel):
    score: str = Field(
        description = "question is about the specified topics? if yes -> Yes if not -> NO"
        
    )



###Question Rewriter agent
def question_rewriter(state: Agentstate):
    print(f"Entering question rewriter agent ")
    current_question = state["question"]
    messages = [
        SystemMessage(
            content = "You are a helpful assistant that rephrases the user's question to be a standalone question optimized for retrieval."
        ),
        HumanMessage(
            content = current_question
        )
    ]
    rephrase_format = ChatPromptTemplate.from_messages(messages)
    response = rephrase_format | llm
    better_question = response.content.strip()

    print(f"Rephased Qusetion: {better_question}")
    state["rephrased_question"] = better_question

    return state


###Question classfier agent 
def question_classifier(state: Agentstate):
    print("Entering Question Classifier Agent")
    
    rephrased_question = state["rephrased_question"]  

    messages = [
        SystemMessage(
            content = """You are a classifier that determines whether a user's question is about one of the following topics

1. Gym History & Founder
2. Operating Hours
3. Membership Plans 
4. Fitness Classes
5. Personal Trainers
6. Facilities & Equipment
7. Anything else about Peak Performance Gym

If the question IS about any of these topics, respond with 'Yes'. Otherwise, respond with 'No'.
"""
        ),
        HumanMessage(
            content = f"User Rephrased Question: {rephrased_question}"
        )
    ]

    grade_format = ChatPromptTemplate.from_messages(messages)
    structured_llm = llm.with_structured_output(GradeQuestion)
    result = grade_format |  structured_llm
    response = result.invoke({})
    state["on_topic"] = response.score.strip()
    print(f"Question classfier: {state["on_topic"]}")
    return state