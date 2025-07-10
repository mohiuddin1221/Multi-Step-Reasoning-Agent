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







## On Topic or Topic Router
def on_topic_router(state: Agentstate):
    print("Entring On topic router")
    on_topic = state.get("on_topic").strip().lower()
    if on_topic == "yes":
        print("Routing to Retrive")
        return "retrieve"
    else:
        print("Routing to off_topic_response")
        return "off_topic_response"
    


### data retrrive form vector databse 
def retrieve(state:Agentstate):
    print("Entirring Retrive")
    rephased_question = state["rephrased_question"]
    documents = retriever.ivoke(rephased_question)
    print(f"retrieve: Retrieved {len(documents)} documents")
    documents = retriever.ivoke(rephased_question)
    state["documents"] = documents
    return state



####retrival grade agent
def retrieval_grader(state:Agentstate):
    system_message = SystemMessage(
        content = """You are a grader assessing the relevance of a retrieved document to a user question.
Only answer with 'Yes' or 'No'.

If the document contains information relevant to the user's question, respond with 'Yes'.
Otherwise, respond with 'No'."""
    )
    structured_llm = llm.with_structured_output(GradeQuestion)
    relevent_docs = []
    for docs in state["documents"]:
        human_message = HumanMessage(
            content = f"User Question: {state["rephrased_question"]} \n\nRetrived Document:\n{docs.page_content}"
        )
        prompt = ChatPromptTemplate.from_message([system_message, human_message])
        result  = prompt | structured_llm
        response = result.invoke({})
        if response.score.strip() == "yes":
            relevent_docs.append(docs)
    state["documents"] = relevent_docs
    state["proceed_to_generate"] = len(relevent_docs) > 0
    print(f"retriveal_grader: {state["proceed_to_generate"]}")
    return state

def proced_router(state:Agentstate):
    rephase_count = state.get("rephase_count", 0)
    if state.get("proceed_to_generate", False):
        print("Routing to generate answer")
        return "generate_answer"
    elif rephase_count >= 2:
        print("Rounting can not anser")
        return "cannot_answer"
    else:
        print("Routing Refine question")
        return "refine_question"
    
def refine_question(state:Agentstate):
    print("Refine agent is running")
    rephase_count = state["rephase_count"]
    if rephase_count > 2:
        print("maximum rephase attempts reached")
        return state
    question_to_refine = state["rephrased_question"]
    system_message = SystemMessage(
        content = """
You are a helpful assistant that slightly refines the user's question to improve retrieval results.
Provide a slightly adjusted version of the question.
"""
    )
    human_message = HumanMessage(
        content = f"Original question: {question_to_refine}\n\nProvide a slightly refined question."
    )
    prompt = ChatPromptTemplate.from_messages([system_message, human_message])
    result = prompt | llm
    response = result.invoke({})
    refined_question = response.content.strip()
    state["rephrased_question"] = refined_question
    state["rephase_count"] = rephase_count+1
    return state



def generate_answer(state:Agentstate):
    print("Generate Answer is running")
    llm = llm
    system_message = SystemMessage(
    content = """
You are an expert assistant designed to generate helpful, accurate, and concise answers based on the provided user question and relevant documents.

Use only the information from the documents to construct your answer. Do not make up or hallucinate any information. If the documents do not contain enough information to answer the question, respond with "I don't know based on the provided documents."

Always aim to provide answers that are clear and directly address the user's question.
"""
)
    rephrased_question = state["rephrased_question"]
    documents = state["documents"]
    doc_text = "\n\n".join([doc.page_content for doc in documents])
    human_message = HumanMessage(
        content=f"User question: {rephrased_question}\n\nRelevant documents:\n{doc_text}"
    )
    prompt = ChatPromptTemplate.from_messages([system_message, human_message])
    result = prompt | llm
    response = result.invoke({})
    state["messages"] = response.content.strip()
    return state

def cannot_answer(state:Agentstate):
    print("Can not answe agent running")
    state["messages"].append(AIMessage(content = "I'm sorry! I cannot answer this question!"))


def off_topic_response(state:Agentstate):
    print("Entering off topic response")
    state["messages"].append(AIMessage(content= "I'm sorry! I cannot answer this question!"))
    return state