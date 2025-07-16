from typing import List, TypedDict
from langchain.schema import Document
from openai import BaseModel
from pydantic import Field
from langchain_core.messages import BaseMessage


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