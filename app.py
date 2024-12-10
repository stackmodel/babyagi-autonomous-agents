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
from langchain_aws import ChatBedrock, BedrockEmbeddings
from dotenv import load_dotenv
import faiss
import os
import logging

# Load environment variables
load_dotenv()

# Setup logging configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def initialize_models() -> tuple:
    """
    Initializes both the embedding model and language model.
    
    Returns:
        tuple: containing the initialized embedding model and language model.
    """
    try:
        embeddings_model = BedrockEmbeddings(credentials_profile_name="default", region_name="us-east-1")
        llm = ChatBedrock(
            model_id="anthropic.claude-3-sonnet-20240229-v1:0",
            credentials_profile_name="default",
            model_kwargs=dict(temperature=0),
        )
        logger.info("Successfully initialized models.")
        return embeddings_model, llm
    except Exception as e:
        logger.error(f"Error initializing models: {e}")
        raise

def initialize_vector_store(embeddings_model) -> FAISS:
    """
    Initializes the FAISS vector store.
    
    Args:
        embeddings_model: The embedding model used to generate vectors.
    
    Returns:
        FAISS: The initialized FAISS vector store.
    """
    try:
        embedding_size = 1536
        index = faiss.IndexFlatL2(embedding_size)
        vectorstore = FAISS(embeddings_model.embed_query, index, InMemoryDocstore({}), {})
        logger.info("Successfully initialized vector store.")
        return vectorstore
    except Exception as e:
        logger.error(f"Error initializing vector store: {e}")
        raise

def initialize_tools(llm_chain: LLMChain) -> list:
    """
    Initializes the tools used by the agent, such as the search tool and todo tool.
    
    Args:
        llm_chain: The LLM chain used for todo creation.
    
    Returns:
        list: A list of initialized tools.
    """
    search = SerpAPIWrapper(serpapi_api_key=os.getenv('SERPAPI_API_KEY'))
    tools = [
        Tool(
            name="Search",
            func=search.run,
            description="Useful for answering questions about current events."
        ),
        Tool(
            name="TODO",
            func=llm_chain.run,
            description="Generates todo lists for a given objective."
        ),
    ]
    return tools

def initialize_agent(llm_chain: LLMChain, tools: list) -> AgentExecutor:
    """
    Initializes the agent with the provided LLM chain and tools.
    
    Args:
        llm_chain: The LLM chain used for the agent.
        tools: List of tools available for the agent.
    
    Returns:
        AgentExecutor: The initialized agent executor.
    """
    tool_names = [tool.name for tool in tools]
    agent = ZeroShotAgent(llm_chain=llm_chain, allowed_tools=tool_names)
    agent_executor = AgentExecutor.from_agent_and_tools(
        agent=agent, tools=tools, handle_parsing_errors=True, verbose=True
    )
    return agent_executor

def run_baby_agi(objective: str, llm: ChatBedrock, vectorstore: FAISS, agent_executor: AgentExecutor) -> None:
    """
    Runs BabyAGI with the given objective, LLM, vectorstore, and agent executor.
    
    Args:
        objective (str): The goal or task to accomplish.
        llm (ChatBedrock): The language model used by BabyAGI.
        vectorstore (FAISS): The vector store used for task tracking.
        agent_executor (AgentExecutor): The agent executor managing task execution.
    """
    try:
        baby_agi = BabyAGI.from_llm(
            llm=llm,
            vectorstore=vectorstore,
            task_execution_chain=agent_executor,
            verbose=False,
            max_iterations=3,
        )
        baby_agi({"objective": objective})
    except Exception as e:
        logger.error(f"Error running BabyAGI: {e}")
        raise

def main():
    """Main function to orchestrate the BabyAGI process."""
    try:
        # Ensure the objective is set in environment variables
        objective = os.getenv('OBJECTIVE')
        if not objective:
            raise ValueError("OBJECTIVE environment variable is not set.")

        # Initialize models and vector store
        embeddings_model, llm = initialize_models()
        vectorstore = initialize_vector_store(embeddings_model)

        # Initialize prompt and LLM chain
        todo_prompt = PromptTemplate.from_template(
            "You are a planner who is an expert at coming up with a todo list for a given objective. Come up with a todo list for this objective: {objective}"
        )
        todo_chain = LLMChain(llm=llm, prompt=todo_prompt)

        # Initialize tools
        tools = initialize_tools(todo_chain)

        # Initialize agent
        agent_executor = initialize_agent(todo_chain, tools)

        # Run BabyAGI with the given objective
        run_baby_agi(objective, llm, vectorstore, agent_executor)

        logger.info("BabyAGI process completed successfully.")
    except Exception as e:
        logger.error(f"Error in main process: {e}")

if __name__ == "__main__":
    main()