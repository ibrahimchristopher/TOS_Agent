from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings
from llama_index.core import VectorStoreIndex
from llama_index.core.objects import ObjectIndex
from llama_index.llms.groq import Groq
from utils import get_doc_tools
from pathlib import Path
from llama_index.core.agent import FunctionCallingAgentWorker
from llama_index.core.agent import AgentRunner
from dotenv import load_dotenv
import os

load_dotenv()




def create_agent():

    api_key = os.environ["api_key"]
    print(api_key)

    #groq_llm = Groq(model="llama-3.3-70b-versatile", api_key=api_key)


    Settings.embed_model = HuggingFaceEmbedding(
        model_name="BAAI/bge-small-en-v1.5"
    )


    papers = [
        'terms/Twitter.pdf',
        'terms/LinkedIn.pdf',
        'terms/TikTok.pdf',
        'terms/Reddit.pdf',
        'terms/Snapchat.pdf',
        'terms/Meta.pdf'
    ]


    paper_to_tools_dict = {}
    for paper in papers:
        print(f"Getting tools for paper: {paper}")
        #vector_tool, summary_tool = get_doc_tools(paper, Path(paper).stem, Settings, groq_llm)
        vector_tool = get_doc_tools(paper, Path(paper).stem, Settings)
        #paper_to_tools_dict[paper] = [vector_tool, summary_tool] #initially 
        paper_to_tools_dict[paper] = [vector_tool]

    all_tools = [t for paper in papers for t in paper_to_tools_dict[paper]]

    llm = Groq(model="llama-3.3-70b-versatile", api_key=api_key)
    obj_index = ObjectIndex.from_objects(
        all_tools,
        index_cls=VectorStoreIndex,
        
    )

    obj_retriever = obj_index.as_retriever(similarity_top_k=2)
    
    
    agent_worker = FunctionCallingAgentWorker.from_tools(
    tool_retriever =obj_retriever,
    llm=llm,
    system_prompt=""" \
    You are a paralegal agent designed to answer queries over a set of given terms of use documents.
    Please always use the tools provided to answer a question. Do not rely on prior knowledge.
    Always support your response with page numbers and file names provided by the context,
    for example: according to page (page_number) on (document name).......\
    """,
        verbose=True
    )
    
    agent = AgentRunner(agent_worker)

    return agent

#agent = create_agent()

#response = agent.query("what is the minimum age for twitter users")

#print(str(response))


