# Load environment variables (API keys, etc.)
import os
from configuration import ConfluenceConnection, create_agent_executor, load_confluence
from colorama import init, Fore, Style
from dotenv import load_dotenv


load_dotenv()

# 1. First, set up your OpenAI API key
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError("OpenAI API key not found. Please set it in your environment variables.")

# 2. Define your Confluence connection parameters
confluence = ConfluenceConnection()
confluence.url = os.getenv("CONFLUENCE_URL")
confluence.username =   os.getenv("CONFLUENCE_USERNAME")
confluence.token =  os.getenv("CONFLUENCE_TOKEN")
confluence.space_key =  os.getenv("CONFLUENCE_SPACE_KEY")
confluence.include_attachments = False
confluence.limit = int(os.getenv("CONFLUENCE_LIMIT", 1000) ) 
confluence.ocr_languages = os.getenv("CONFLUENCE_OCR_LANGUAGES")  
confluence.max_pages = int(os.getenv("CONFLUENCE_MAX_PAGES", 10000))

# 3. Load data from Confluence into Milvus
load_confluence(confluence)


if not confluence.url or not confluence.username or not confluence.token:
    raise ValueError("Confluence connection parameters are not set properly.")


# Initialize colorama (required for Windows)
init()

# 4. Create the agent executor
agent_executor = create_agent_executor(
    openai_api_key=openai_api_key,
    model_name="gpt-4o"
)

# 5. Start an interactive chat loop with colors
print(f"\n{Fore.CYAN}=== Confluence Knowledge Base Agent ==={Style.RESET_ALL}")
print(f"{Fore.YELLOW}Type 'exit' or 'quit' to end the conversation.{Style.RESET_ALL}\n")


chat_history = []

while True:
    user_input = input("\nYou: ")
    
    # Check if user wants to exit
    if user_input.lower() in ["exit", "quit", "bye"]:
        print("Goodbye! The agent has been closed.")
        break
    
    # Get response from agent
    result = agent_executor.invoke({
        "input": user_input,
        "chat_history": chat_history
    })
    
    # Print the agent's response
    print(f"\n{Fore.BLUE}Agent: {Style.RESET_ALL}{result['output']}")    
    
    # Add the interaction to chat history
    chat_history.append({"role": "user", "content": user_input})
    chat_history.append({"role": "assistant", "content": result['output']})