import logging
from typing import List, Dict, Any, Optional

from evolving_agents.core.llm_service import LLMService
# from evolving_agents.core.smart_context import Message # Using Dict for messages as per instruction

class MessageSummarizationTool:
    """
    Summarizes a list of messages focusing on relevance to a target goal using an LLM.
    """
    name: str = "message_summarization_tool"
    description: str = (
        "Summarizes a list of messages focusing on relevance to a target goal using an LLM."
    )

    def __init__(self, llm_service: LLMService):
        """
        Initializes the MessageSummarizationTool.

        Args:
            llm_service: An instance of LLMService for language model interactions.
        """
        self.llm_service = llm_service
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    async def summarize_messages(
        self,
        messages: List[Dict[str, Any]],
        target_goal: str,
        max_summary_tokens: int = 250
    ) -> str:
        """
        Summarizes a list of messages, focusing on aspects relevant to a target goal.

        Args:
            messages: A list of message dictionaries. Each dictionary should at least
                      contain 'sender_id' and 'content'.
            target_goal: A string describing the goal to focus the summary on.
            max_summary_tokens: An approximate guide for the desired length of the summary in tokens.

        Returns:
            A string containing the generated summary.
        """
        if not messages:
            self.logger.info("No messages provided to summarize.")
            return "No messages to summarize."

        formatted_conversation = ""
        for msg in messages:
            sender = msg.get("sender_id", "Unknown Sender")
            content = msg.get("content", "")
            formatted_conversation += f"Sender: {sender}\nMessage: {content}\n\n---\n"

        # Estimate words based on tokens (very rough estimate, e.g., 1 token ~ 0.75 words)
        # This is just to give the LLM a rough idea. The actual token limit is handled by the LLM service if configured.
        approx_words = int(max_summary_tokens * 0.75)

        prompt = f"""Given the following conversation history:

--- BEGIN CONVERSATION ---
{formatted_conversation.strip()}
--- END CONVERSATION ---

Considering the target goal: "{target_goal}"

Please provide a concise summary of the conversation. The summary should:
1. Focus on aspects most relevant to achieving the stated target goal.
2. Highlight key information, requirements, decisions made, and any outstanding questions or critical information needed to achieve the target goal.
3. Be clear and to the point.
4. Aim for a length of approximately {approx_words} words.

Summary:
"""

        try:
            self.logger.info(
                f"Summarizing {len(messages)} messages with target goal: '{target_goal}' "
                f"(max_summary_tokens: {max_summary_tokens})."
            )
            # Assuming llm_service.generate() can take max_tokens or similar
            # and handles the actual token limiting.
            # The prompt guides the LLM on length, but the service might enforce it.
            summary = await self.llm_service.generate(
                prompt,
                # max_tokens=max_summary_tokens # Pass max_tokens if your LLMService supports it
            )
            self.logger.info("Successfully generated message summary.")
            return summary.strip()
        except Exception as e:
            self.logger.error(f"Error during message summarization: {e}")
            return f"Error: Could not generate summary due to: {e}"

# Example Usage (Conceptual - would typically be part of a larger system)
async def example_usage():
    # This is a mock LLMService for demonstration.
    # In a real scenario, this would be a fully implemented LLMService.
    class MockLLMService(LLMService):
        async def generate(self, prompt: str, **kwargs) -> str:
            # Simulate LLM response based on prompt
            if "invoice processing feature" in prompt:
                return (
                    "The conversation focused on developing an invoice processing feature. "
                    "Key requirements include extracting vendor name, invoice date, and total amount "
                    "from PDF and scanned image formats. OCR is needed for scanned images. "
                    "An outstanding question is the choice of OCR tool."
                )
            return "Default summary based on prompt."

        async def embed(self, text: str) -> List[float]:
            return [0.1] * 768 # Dummy embedding

    llm_service_instance = MockLLMService(api_key="dummy_key", default_model="dummy_model")
    summarizer = MessageSummarizationTool(llm_service=llm_service_instance)

    sample_messages = [
        {"sender_id": "user_A", "content": "I need to develop a new feature for processing invoices."},
        {"sender_id": "agent_B", "content": "Okay, what are the key requirements for this invoice processing feature?"},
        {"sender_id": "user_A", "content": "It needs to extract vendor name, invoice date, and total amount. It should also handle PDF and scanned image formats."},
        {"sender_id": "agent_B", "content": "Understood. For scanned images, OCR will be necessary. Do we have a preferred OCR tool?"}
    ]
    goal = "Determine the necessary tools and steps to implement the invoice processing feature."

    summary_result = await summarizer.summarize_messages(sample_messages, goal, max_summary_tokens=100)
    print("--- Example Usage ---")
    print(f"Target Goal: {goal}")
    print(f"Generated Summary:\n{summary_result}")

if __name__ == "__main__":
    import asyncio
    # To run the example:
    # Ensure you have a concrete LLMService or a mock like above.
    # If LLMService constructor needs more params, adjust MockLLMService.
    # asyncio.run(example_usage())
    pass # Comment out pass and uncomment asyncio.run to test example if LLMService is concrete.
