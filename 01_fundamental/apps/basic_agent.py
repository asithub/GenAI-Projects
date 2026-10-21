import os
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.tools import tool

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
model_name = os.getenv("OPENAI_MODEL", "gpt-5-nano")
if not api_key:
    raise ValueError("OPENAI_API_KEY is missing from your .env file")

llm = ChatOpenAI(
    model = model_name,
    temperature = 0,
    api_key = api_key,
    timeout=30,
    max_retries=2
)

@tool("addition_tool", description="This tool is for adding the number")
def add_number(a:float, b:float) -> float:
    """
        rules:
        - It will return the sum of two numbers

        Args:
            a: Number
            b: Number 
    """
    return a + b

agent = create_agent(
    model=llm,
    tools=[add_number],
    system_prompt="You are a helpful assistant"
)

response = agent.invoke(
    {"messages": 
     [
         {
             "role": "user", 
             "content": "What's 2 + 2"
          }
      ]}
)
print(response["messages"][-1].content)


