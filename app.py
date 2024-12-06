"""
Author: stackmodel

Credits:
This project is built upon the following resources and contributions:
  -  BabyAGI by Yohei Nakajima: The original concept and implementation of BabyAGI, framework for a self-building autonomous agent, was developed by Yohei Nakajima. 
     You can find more about the original project here -> (https://github.com/yoheinakajima/babyagi).
  -  LangChain: This project utilizes LangChain, a powerful framework for building language model-powered applications.
     You can find more information here (https://github.com/langchain-ai/langchain) and the 
     cookbook [here](https://github.com/langchain-ai/langchain/blob/master/cookbook/baby_agi_with_agent.ipynb).

"""
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_experimental.autonomous_agents import BabyAGI
from langchain.docstore import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from langchain.agents import AgentExecutor, Tool, ZeroShotAgent
from langchain_community.utilities import SerpAPIWrapper
from langchain_aws import BedrockLLM, ChatBedrock
from langchain_aws import BedrockEmbeddings
from dotenv import load_dotenv
import faiss
import os

# Load environment variables
load_dotenv()

#Function to initialize both embedding model and LLM
def initialize_models():
    """Initializes the embedding model and language model."""
    embeddings_model = BedrockEmbeddings(credentials_profile_name="default", region_name="us-east-1")

    # using Anthropic
    llm = ChatBedrock(
        model_id="anthropic.claude-3-sonnet-20240229-v1:0",
        credentials_profile_name="default",
        model_kwargs=dict(temperature=0),
    )
    
    return embeddings_model, llm

# Function to initialize the vector store
def vector_store(embeddings_model):
    """Resets the vector store."""
    embedding_size = 1536
    index = faiss.IndexFlatL2(embedding_size)
    vectorstore = FAISS(embeddings_model.embed_query, index, InMemoryDocstore({}), {})
    return vectorstore

# Initialize models and vector store
embeddings_model, llm = initialize_models()
vectorstore = vector_store(embeddings_model)

todo_prompt = PromptTemplate.from_template(
    "You are a planner who is an expert at coming up with a todo list for a given objective. Come up with a todo list for this objective: {objective}"
)
todo_chain = LLMChain(llm=llm, prompt=todo_prompt)
search = SerpAPIWrapper(serpapi_api_key=os.getenv('SERPAPI_API_KEY'))
tools = [
    Tool(
        name="Search",
        func=search.run,
        description="useful for when you need to answer questions about current events",
    ),
    Tool(
        name="TODO",
        func=todo_chain.run,
        description="useful for when you need to come up with todo lists. Input: an objective to create a todo list for. Output: a todo list for that objective. Please be very clear what the objective is!",
    ),
]

prefix = """You are an AI who performs one task based on the following objective: {objective}. Take into account these previously completed tasks: {context}."""
suffix = """Question: {task}
{agent_scratchpad}"""
prompt = ZeroShotAgent.create_prompt(
    tools,
    prefix=prefix,
    suffix=suffix,
    input_variables=["objective", "task", "context", "agent_scratchpad"],
)

llm_chain = LLMChain(llm=llm, prompt=prompt)
tool_names = [tool.name for tool in tools]
agent = ZeroShotAgent(llm_chain=llm_chain, allowed_tools=tool_names)
agent_executor = AgentExecutor.from_agent_and_tools(
    agent=agent, tools=tools, handle_parsing_errors=True, verbose=True
)

OBJECTIVE = os.getenv('OBJECTIVE')
# Logging of LLMChains
verbose = False
# If None, will keep on going forever
max_iterations = 3
baby_agi = BabyAGI.from_llm(
    llm=llm,
    vectorstore=vectorstore,
    task_execution_chain=agent_executor,
    verbose=verbose,
    max_iterations=max_iterations,
)

baby_agi({"objective": OBJECTIVE})