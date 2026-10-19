#!/usr/bin/env python3
#pip install -r requirements.txt
#pip install --upgrade "strands-agents-tools[mem0_memory]"
"""
Personal AI Agent using Strands, Ollama, Mem0, and web search.

Capabilities:
- Store personal information and preferences
- Retrieve relevant memories
- List saved memories
- Search the web
- Answer questions using the LLM
"""

import os

from ddgs import DDGS
from ddgs.exceptions import DDGSException, RatelimitException
from strands import Agent, tool
from strands.models.ollama import OllamaModel
from strands_tools import mem0_memory


# --------------------------------------------------
# 1. Configure Mem0
# --------------------------------------------------

USER_ID = "Sneha Augustine"

# Set your replacement MEM0_API_KEY in the environment
if not os.getenv("MEM0_API_KEY"):
    raise RuntimeError(
        "MEM0_API_KEY is missing. Set it in your environment first."
    )

os.environ["MEM0_USER_ID"] = USER_ID
# --------------------------------------------------
# 2. Configure the local Ollama model
# --------------------------------------------------

model = OllamaModel(
    host="http://localhost:11434",
    model_id="llama3.1",
)


# --------------------------------------------------
# 3. Define the web search tool
# --------------------------------------------------

@tool
def websearch(
    keywords: str,
    region: str = "us-en",
    max_results: int = 5,
) -> str:
    """Search the web for current information.

    Args:
        keywords: Search query.
        region: Search region, e.g. us-en or uk-en.
        max_results: Maximum number of results.

    Returns:
        Search results formatted as text.
    """
    try:
        results = DDGS().text(
            keywords,
            region=region,
            max_results=max_results,
        )

        if not results:
            return "No search results found."

        return "\n\n".join(
            f"Title: {item.get('title', '')}\n"
            f"URL: {item.get('href', '')}\n"
            f"Summary: {item.get('body', '')}"
            for item in results
        )

    except RatelimitException:
        return "Web search rate limit reached. Please try again later."

    except DDGSException as error:
        return f"Web search error: {error}"


# --------------------------------------------------
# 4. Define the system prompt
# --------------------------------------------------

SYSTEM_PROMPT = """
You are a helpful personal AI assistant.

You have access to:
- mem0_memory for storing and retrieving user memories
- websearch for finding current information online

Instructions:
1. Retrieve relevant memories before answering personal questions.
2. Store new personal facts and preferences when the user asks
   you to remember them or shares useful information.
3. Use websearch when current or external information is needed.
4. Answer naturally and conversationally.
5. Never claim a memory was saved unless the tool indicates success.
6. If a memory operation reports an error, be transparent about it.
"""


# --------------------------------------------------
# 5. Create the agent
# --------------------------------------------------

memory_agent = Agent(
    system_prompt=SYSTEM_PROMPT,
    model=model,
    tools=[mem0_memory, websearch],
)


# --------------------------------------------------
# 6. Run the interactive agent
# --------------------------------------------------

def main():
    print("\n🧠 Personal AI Agent 🧠")
    print("Using Ollama + Mem0 + Web Search")
    print("Commands: 'memory' to list memories, 'exit' to quit.")
    print("Configured model:", type(model).__name__)
    print("Model details:", model)
    while True:
        try:
            user_input = input("\nYou: ").strip()

            if not user_input:
                continue

            if user_input.lower() == "exit":
                print("Goodbye!")
                break

            elif user_input.lower() == "memory":
                result = memory_agent.tool.mem0_memory(
                    action="list"
                )
                print("\nSaved memories:")
                print(result)

            else:
                response = memory_agent(user_input)
                print("\nAssistant:")
                print(response)

        except KeyboardInterrupt:
            print("\nGoodbye!")
            break

        except Exception as error:
            print(f"\nAn error occurred: {error}")


if __name__ == "__main__":
    main()
