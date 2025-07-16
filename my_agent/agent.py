
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import AzureChatOpenAI
from langgraph.graph import StateGraph, START, END
from utils.state import Agentstate
from utils.nodes import(
    question_rewriter,
    question_classifier,
    retrieve, 
    retrieval_grader,
    refine_question,
    generate_answer,
    cannot_answer,
    off_topic_response,
    on_topic_routerc,
    retrieval_grader,
    on_topic_router,
    retrieval_grader,
    proced_router,
    generate_answer,
    cannot_answer,
    off_topic_response




)




#####Workflow#####
workflow = StateGraph(Agentstate)
workflow.add_node("question_rewriter", question_rewriter)
workflow.add_node("question_classifier", question_classifier)

workflow.add_node("retrieve", retrieve)
workflow.add_node("retrieval_grader", retrieval_grader)
workflow.add_node("refine_question", refine_question)
workflow.add_node("generate_answer", generate_answer)
workflow.add_node("cannot_answer", cannot_answer)
workflow.add_node("off_topic_response", off_topic_response)

###worflow edge
workflow.add_edge(START, "question_rewriter")
workflow.add_edge("question_rewriter", "question_classifier")
workflow.add_conditional_edges(
    "question_classifier",
    on_topic_router,
    {
        "retrieve", retrieve,
        "off_topic_response", off_topic_response
    }


)
workflow.add_edge("retrieve", "retrieval_grader")
workflow.add_conditional_edges(
    "retrieval_grader",
    proced_router,
    {
        "refine_question", refine_question,
        "generate_answer", generate_answer,
        "cannot_answer", cannot_answer

    }
)

workflow.add_edge("refine_question", "retrieve")
workflow.add_edge("generate_answer", END)
workflow.add_edge("cannot_answer", END)
workflow.add_edge("off_topic_response", END)


graph = workflow.compile()