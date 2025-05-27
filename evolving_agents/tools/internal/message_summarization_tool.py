import logging
from typing import List, Dict, Any, Optional

from evolving_agents.core.base import BaseTool
from evolving_agents.core.llm_service import LLMService

# Configure logging
logger = logging.getLogger(__name__)

class MessageSummarizationTool(BaseTool):
    """
    A tool to summarize a list of messages using an LLM,
    focusing on aspects relevant to a specified target goal.
    """
    name: str = "MessageSummarizationTool"
    description: str = (
        "Summarizes a list of messages (e.g., conversation history) "
        "with a focus on a given target goal, using an LLM."
    )

    def __init__(self, llm_service: LLMService):
        """
        Initializes the MessageSummarizationTool.

        Args:
            llm_service: An instance of LLMService for interacting with the language model.
        """
        super().__init__()
        if llm_service is None:
            raise ValueError("llm_service cannot be None for MessageSummarizationTool")
        self.llm_service = llm_service

    async def summarize_messages(
        self,
        messages: List[Dict[str, Any]],
        target_goal: str,
        max_summary_length: int = 150, # Approximate target in words
    ) -> str:
        """
        Summarizes a list of messages focusing on a target goal.

        Args:
            messages: A list of message dictionaries. Each dictionary is expected
                      to have at least 'sender_id' (or 'sender', 'user') and 'content' keys.
            target_goal: The goal to guide the summarization process.
            max_summary_length: Approximate desired maximum length of the summary in words.
                                LLMs handle length constraints contextually.

        Returns:
            A string containing the summary, or an empty string if an error occurs
            or no summary can be generated.
        """
        if not messages:
            logger.warning("summarize_messages called with no messages.")
            return "No messages provided to summarize."
        if not target_goal:
            logger.warning("summarize_messages called with no target_goal.")
            return "No target goal provided for summarization."

        formatted_messages = []
        for msg in messages:
            sender = msg.get("sender_id", msg.get("sender", msg.get("user", "Unknown Sender")))
            content = msg.get("content", "")
            if content: # Only include messages with content
                formatted_messages.append(f"{sender}: {content}")
        
        if not formatted_messages:
            logger.info("No messages with content found to summarize.")
            return "No message content available to summarize."

        messages_string = "\n".join(formatted_messages)

        prompt = f"""Here is a list of messages:
{messages_string}

The current goal is: "{target_goal}"

Please provide a concise summary of these messages. The summary should focus on information relevant to achieving or understanding progress towards this goal. Aim for a summary of around {max_summary_length} words.
Summary:"""

        try:
            # Assuming llm_service.generate() is an async method that takes a prompt string
            # and returns the generated text.
            # Adjust parameters like max_tokens if your LLMService interface supports them
            # to better control summary length.
            summary_text = await self.llm_service.generate(
                prompt,
                # max_tokens can be roughly estimated from max_summary_length,
                # e.g., max_summary_length * 1.5 or 2, depending on average token length
                # This is highly model-dependent.
                # For now, we assume the prompt's instruction is the primary length guide.
            )
            if not summary_text:
                logger.warning(f"LLM returned an empty summary for goal: {target_goal}")
                return "Summary generation resulted in an empty response."
            return summary_text.strip()
        except Exception as e:
            logger.error(f"Error during LLM call for message summarization: {e}", exc_info=True)
            return f"Error summarizing messages: {e}"

    # Example of how BaseTool's execute might be structured (if needed by your framework)
    # async def execute(self, messages: List[Dict[str, Any]], target_goal: str, max_summary_length: int = 150) -> str:
    #     """
    #     Executes the message summarization.
    #     """
    #     if not isinstance(messages, list):
    #         raise ValueError("Input 'messages' must be a list of dictionaries.")
    #     if not all(isinstance(msg, dict) for msg in messages):
    #         raise ValueError("Each item in 'messages' must be a dictionary.")
    #     if not isinstance(target_goal, str):
    #         raise ValueError("Input 'target_goal' must be a string.")
    #     if not isinstance(max_summary_length, int) or max_summary_length <= 0:
    #         raise ValueError("'max_summary_length' must be a positive integer.")
            
    #     return await self.summarize_messages(
    #         messages=messages,
    #         target_goal=target_goal,
    #         max_summary_length=max_summary_length
    #     )

# Example Usage (Illustrative - requires async environment and mock/real services)
# async def main():
#     class MockLLMService:
#         async def generate(self, prompt: str, **kwargs) -> str:
#             print(f"\n--- LLM Prompt for Summarization ---\n{prompt}\n--- End Prompt ---")
#             # Simulate LLM response based on prompt
#             if "find a good Italian restaurant" in prompt:
#                 return "The group discussed preferences for Italian food, mentioning pasta and pizza, and decided to look for places near downtown."
#             return "This is a generic summary of the provided messages."

#     llm_service = MockLLMService()
#     summarization_tool = MessageSummarizationTool(llm_service=llm_service)

#     sample_messages = [
#         {"sender_id": "Alice", "content": "Hey Bob, are we still on for dinner tonight?"},
#         {"sender_id": "Bob", "content": "Yes! I was thinking Italian. What do you think?"},
#         {"sender_id": "Alice", "content": "Sounds great! I love pasta. Any specific place in mind?"},
#         {"sender_id": "Bob", "content": "Not yet, maybe something downtown? We should check reviews."},
#         {"sender_id": "Charlie", "content": "I can join too if it's Italian!"}, # Unrelated sender, but part of conversation
#         {"sender_id": "Alice", "content": "Okay, let's find a good Italian restaurant downtown."}
#     ]

#     target_goal_1 = "Arrange dinner plans, specifically to find a suitable Italian restaurant."
#     summary1 = await summarization_tool.summarize_messages(sample_messages, target_goal_1, max_summary_length=50)
#     print(f"\nSummary for Goal 1 (approx 50 words):\n{summary1}")

#     target_goal_2 = "Determine Bob's availability for dinner."
#     summary2 = await summarization_tool.summarize_messages(sample_messages, target_goal_2, max_summary_length=30)
#     print(f"\nSummary for Goal 2 (approx 30 words):\n{summary2}")
    
#     empty_messages = []
#     summary_empty = await summarization_tool.summarize_messages(empty_messages, target_goal_1)
#     print(f"\nSummary for empty messages:\n{summary_empty}")

# if __name__ == "__main__":
#     import asyncio
#     asyncio.run(main())
#     pass
