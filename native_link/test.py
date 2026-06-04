from supervisor import Reporter
import asyncio

async def main():
    r = Reporter()
    r.purge_all()
    
    full_response = ""
    async for chunk in r("use orchestrator tool to list all files in cwd","1"):

        
        full_response += chunk
    print("*******************************************************")
    return full_response


if __name__ == "__main__":
   
    print(asyncio.run(main()))
# from langchain.agents import create_agent
# from langchain_openai import ChatOpenAI

# core_model = ChatOpenAI(
#             model= "qwen3:4b-instruct",
#             base_url="http://localhost:11434/v1",
#             temperature=0.2,
#             api_key="password", 
            
#         ) 
  
# agent = create_agent(
#     core_model, 
# )

# for i in agent.stream()