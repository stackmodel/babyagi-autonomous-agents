**Build Your Own Autonomous Research Companion**

**Overview**

This repository was created after exploring the [BabyAGI](https://github.com/yoheinakajima/babyagi) project by Yohei Nakajima and studying the [LangChain cookbook](https://github.com/langchain-ai/langchain/blob/master/cookbook/baby_agi.ipynb?ref=blog.langchain.dev), which demonstrates how easily different models and vector stores can be swapped out. 

In this implementation, I made few improvements:
  -  Integrated Amazon Bedrock to leverage its benefits, such as access to foundation models, model customization, and improved privacy.
  -  Swapped the OpenAI model with the Anthropic Claude Sonnet model.
  -  Simplified the process by allowing the **objective** to be controlled through the .env file, eliminating the need to modify the code directly.
  -  Moved away from the notebook-based approach to a more structured Python script.

**Prerequisites**

Before running the code, make sure to complete the following:
  - [AWS CLI Setup](https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html): Install and configure the AWS Command Line Interface (CLI) with the appropriate credentials.
  - AWS Bedrock Access: You must have access to Amazon Bedrock and the appropriate credentials to use its models, including the Anthropic Claude Sonnet model.
  - SerpAPI Setup: You need to have a valid SerpAPI API key to access the search functionality. You can get it from [SerpAPI](https://serpapi.com/).
  - Environment Variables: Create a .env file in the root of your project to store sensitive data like your SerpAPI API key and the objective for the BabyAGI agent.
  - Environment Variables: Rename .env.example to .env file in the root of your project to store your SerpAPI API key and objective for the BabyAGI agent.
    Example `.env` file:

```.env
#.env

SERPAPI_API_KEY=your_serpapi_api_key_here
OBJECTIVE='Write a weather report for SF today'
```
**Running the Code**

To run the code, follow the steps below:

- Clone the repo and install Dependencies:

    - Make sure you have Python 3.7 or higher installed. Then, create a virtual environment and install the dependencies:
      
      ```
      python -m venv env
      source env/bin/activate  # For Linux/macOS
      .\env\Scripts\activate   # For Windows
      pip install -r requirements.txt
      ```
- Run the Code:

  -  Once the dependencies are installed and environment variables are set, you can run the app.py script. Execute the following command:
    
      ```python app.py```
     
This will initialize the models, create the vector store, set up the agent with tools (Search, TODO), and then use the BabyAGI agent to execute the objective specified in the .env file.

**Notes**
-  The code is designed to run for up to a maximum of 3 iterations (max_iterations=3) by default. You can adjust this based on your needs.
-  The code uses AWS's Bedrock service, so ensure that your AWS credentials are configured correctly.

**Credits & References** 

This project is built upon the following resources and contributions:

  -  BabyAGI by Yohei Nakajima: The original concept and implementation of BabyAGI, framework for a self-building autonomous agent, was developed by Yohei Nakajima. You can find more about the original project [here](https://github.com/yoheinakajima/babyagi).
  -  LangChain: This project utilizes LangChain, a powerful framework for building language model-powered applications. You can find more information [here](https://github.com/langchain-ai/langchain) and the cookbook [here](https://github.com/langchain-ai/langchain/blob/master/cookbook/baby_agi.ipynb?ref=blog.langchain.dev).






