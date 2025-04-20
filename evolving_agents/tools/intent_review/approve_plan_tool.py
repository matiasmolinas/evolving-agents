# evolving_agents/tools/intent_review/approve_plan_tool.py

import json
import asyncio
import os
import time
from datetime import datetime
import logging
from typing import Dict, Any, List, Optional

from pydantic import BaseModel, Field

# BeeAI Framework imports
from beeai_framework.tools.tool import Tool, StringToolOutput
from beeai_framework.context import RunContext # Keep RunContext import
from beeai_framework.emitter.emitter import Emitter

# Project-specific imports
from evolving_agents.core.llm_service import LLMService
# No longer need Intent* imports here if we pass JSON string
# from evolving_agents.core.intent_review import IntentPlan, Intent, IntentStatus

logger = logging.getLogger(__name__)

class ApprovePlanInput(BaseModel):
    """Input schema for the ApprovePlanTool."""
    plan_id: str = Field(description="ID of the plan to review")
    interactive_mode: bool = Field(True, description="Whether to use interactive mode for review")
    use_agent_reviewer: bool = Field(False, description="Whether to use AI agent to review instead of human")
    agent_prompt: Optional[str] = Field(None, description="Custom prompt for the AI reviewer")
    output_path: Optional[str] = Field(None, description="Path to save the intent plan")

class ApprovePlanTool(Tool[ApprovePlanInput, None, StringToolOutput]):
    """
    Tool for reviewing and approving intent plans before execution.
    Implements the human-in-the-loop workflow for SystemAgent.
    """
    name = "ApprovePlanTool"
    description = "Review and approve intent plans before execution in the SystemAgent"
    input_schema = ApprovePlanInput

    def __init__(
        self,
        llm_service: Optional[LLMService] = None,
        options: Optional[Dict[str, Any]] = None
    ):
        super().__init__(options=options or {})
        self.llm_service = llm_service

    def _create_emitter(self) -> Emitter:
        return Emitter.root().child(
            namespace=["tool", "intent_review", "approve_plan"],
            creator=self,
        )

    async def _run(self, tool_input: ApprovePlanInput, options: Optional[Dict[str, Any]] = None,
                  context: Optional[RunContext] = None) -> StringToolOutput: # Renamed 'input' to 'tool_input'
        """
        Review and approve an intent plan.
        """
        # --- CORRECTED CONTEXT RETRIEVAL ---
        intent_plan_json = None
        if context and hasattr(context, 'context') and isinstance(context.context, dict):
            # Retrieve from the context dictionary
            intent_plan_json = context.context.get("intent_plan")
        # --- END CORRECTION ---

        if not intent_plan_json:
            return StringToolOutput(json.dumps({
                "status": "error",
                "message": "No intent plan JSON found in context"
            }, indent=2))

        try:
            # Parse the plan *after* retrieving it
            intent_plan = json.loads(intent_plan_json)
        except json.JSONDecodeError:
            return StringToolOutput(json.dumps({
                "status": "error",
                "message": "Invalid intent plan JSON in context"
            }, indent=2))

        # Save the intent plan if requested
        if tool_input.output_path: # Use 'tool_input'
            # Ensure directory exists before writing
            output_dir = os.path.dirname(tool_input.output_path)
            if output_dir and not os.path.exists(output_dir):
                 os.makedirs(output_dir, exist_ok=True)

            try:
                with open(tool_input.output_path, 'w') as f:
                    # Write the already parsed dictionary back to JSON
                    json.dump(intent_plan, f, indent=2)
                logger.info(f"Intent plan saved to {tool_input.output_path}")
            except Exception as e:
                 logger.error(f"Failed to save intent plan to {tool_input.output_path}: {e}")


        # Check if we should use AI agent for review
        if tool_input.use_agent_reviewer and self.llm_service: # Use 'tool_input'
            return await self._agent_review_plan(intent_plan, tool_input.agent_prompt) # Use 'tool_input'

        # Human review process
        if tool_input.interactive_mode: # Use 'tool_input'
            return await self._interactive_review_plan(intent_plan)
        else:
            return await self._cli_review_plan(intent_plan)

    async def _agent_review_plan(self, intent_plan: Dict[str, Any], custom_prompt: Optional[str] = None) -> StringToolOutput:
        """Use AI to review the intent plan."""
        if not self.llm_service:
            return StringToolOutput(json.dumps({
                "status": "error",
                "message": "LLM Service not available for AI review"
            }, indent=2))

        # Prepare the review prompt
        default_prompt = f"""
        As an AI Safety Inspector, review the following intent plan for a task.

        Intent Plan:
        ```json
        {json.dumps(intent_plan, indent=2)}
        ```

        For each intent in the plan, evaluate:
        1. Safety: Could this action cause harm or violate ethical guidelines?
        2. Appropriateness: Is this action relevant to the stated objective?
        3. Parameters: Are the parameters complete and valid?
        4. Dependencies: Are the dependencies correctly specified?

        After reviewing, decide whether to:
        1. APPROVE the plan if it appears safe and appropriate
        2. REJECT the plan if you identify safety issues or other problems

        Return your response in this exact JSON format:
        {{
            "status": "approved" or "rejected",
            "reason": "Brief explanation of your decision",
            "intent_reviews": [
                {{
                    "intent_id": "ID of the intent",
                    "status": "safe" or "unsafe",
                    "concerns": ["list", "of", "concerns"] or []
                }}
            ],
            "overall_risk_assessment": "Low/Medium/High"
        }}
        """

        prompt = custom_prompt or default_prompt

        # Get the review from the LLM
        review_text = await self.llm_service.generate(prompt)

        # Try to parse the result as JSON
        try:
            # Extract JSON from the response if needed
            json_start = review_text.find('{')
            json_end = review_text.rfind('}')
            if json_start >= 0 and json_end > json_start:
                review_json = json.loads(review_text[json_start:json_end+1])
            else:
                review_json = json.loads(review_text)

            # Add AI review source
            review_json["review_source"] = "ai"
            return StringToolOutput(json.dumps(review_json, indent=2))
        except json.JSONDecodeError:
            # If parsing fails, return a structured error
            return StringToolOutput(json.dumps({
                "status": "error",
                "message": "Failed to parse AI reviewer response",
                "raw_response": review_text
            }, indent=2))

    async def _interactive_review_plan(self, intent_plan: Dict[str, Any]) -> StringToolOutput:
        """Interactive console-based review of the intent plan."""
        print("\n" + "="*60)
        print("🔍 INTENT PLAN REVIEW 🔍")
        print("="*60)

        print(f"\nTask: {intent_plan.get('objective', intent_plan.get('description', 'Unknown task'))}")
        print(f"Plan ID: {intent_plan.get('plan_id', 'Unknown')}")

        # Display intents
        intents = intent_plan.get("intents", [])
        print(f"\nThe plan contains {len(intents)} intents to execute:")

        for i, intent in enumerate(intents, 1):
            component_name = intent.get("component_name", "Unknown")
            action = intent.get("action", "Unknown")
            step_type = intent.get("step_type", "Unknown")

            if step_type == "RETURN":
                print(f"\n{i}. [{step_type}] Return result")
            else:
                print(f"\n{i}. [{step_type}] {component_name}.{action}")

            # Show parameters
            params = intent.get("params", {})
            if params:
                print(f"   Parameters:")
                # Safely print parameters, handling potential non-string values
                try:
                     param_str = json.dumps(params, indent=6) # Pretty print params
                     print(f"     {param_str}")
                except TypeError:
                     print(f"     (Parameters contain non-serializable data)")


            # Show justification
            justification = intent.get("justification", "No justification provided")
            print(f"   Justification: {justification}")

            # Show dependencies
            dependencies = intent.get("depends_on", [])
            if dependencies:
                print(f"   Depends on: {', '.join(dependencies)}")

        # Ask for decision
        print("\nPlease review this intent plan.")
        print("Does this plan meet the requirements and appear safe to execute?")

        while True:
            choice = input("\nApprove this intent plan? (y/n/d for details/i for intent details): ").strip().lower()

            if choice == 'd':
                # Show full plan details
                print("\nDetailed Intent Plan:")
                print(json.dumps(intent_plan, indent=2))
                continue

            elif choice == 'i':
                # Let user inspect a specific intent
                try:
                    intent_num_str = input("Enter intent number to inspect: ")
                    if not intent_num_str.isdigit():
                        print("Please enter a valid number.")
                        continue
                    intent_num = int(intent_num_str)
                    if 1 <= intent_num <= len(intents):
                        print("\nDetailed Intent Information:")
                        print(json.dumps(intents[intent_num-1], indent=2))
                    else:
                        print(f"Invalid intent number. Must be between 1 and {len(intents)}.")
                except ValueError:
                    print("Please enter a valid number.")
                continue

            elif choice == 'y':
                comments = input("Optional comments or suggestions: ").strip()

                # Mark all intents as approved IN THE LOCAL COPY
                for intent in intents:
                    intent["status"] = "APPROVED"

                intent_plan["status"] = "APPROVED"
                intent_plan["review_timestamp"] = datetime.utcnow().isoformat()
                intent_plan["reviewer_comments"] = comments

                # Important: Return the modified intent plan dictionary
                return StringToolOutput(json.dumps({
                    "status": "approved",
                    "message": "Intent plan approved by reviewer",
                    "comments": comments,
                    "approved_plan": intent_plan # Send back the modified plan
                }, indent=2))

            elif choice == 'n':
                reason = input("Reason for rejection: ").strip()

                # Mark all intents as rejected IN THE LOCAL COPY
                for intent in intents:
                    intent["status"] = "REJECTED"

                intent_plan["status"] = "REJECTED"
                intent_plan["review_timestamp"] = datetime.utcnow().isoformat()
                intent_plan["rejection_reason"] = reason

                return StringToolOutput(json.dumps({
                    "status": "rejected",
                    "message": "Intent plan rejected by reviewer",
                    "reason": reason
                    # Optionally include the rejected plan: "rejected_plan": intent_plan
                }, indent=2))

            else:
                print("Invalid choice. Please enter 'y' (approve), 'n' (reject), 'd' (details), or 'i' (intent details).")

    async def _cli_review_plan(self, intent_plan: Dict[str, Any]) -> StringToolOutput:
        """Non-interactive CLI review of the intent plan."""
        # (Implementation remains largely the same as before, but ensure it uses the parsed dict)
        print("\n" + "="*60)
        print("INTENT PLAN REVIEW (CLI MODE)")
        print("="*60)

        print(f"\nTask: {intent_plan.get('objective', intent_plan.get('description', 'Unknown task'))}")
        print(f"Plan ID: {intent_plan.get('plan_id', 'Unknown')}")

        # Display intents
        intents = intent_plan.get("intents", [])
        print(f"\nThe plan contains {len(intents)} intents to execute:")

        for i, intent in enumerate(intents, 1):
            component_name = intent.get("component_name", "Unknown")
            action = intent.get("action", "Unknown")
            step_type = intent.get("step_type", "Unknown")

            print(f"{i}. [{step_type}] {component_name}.{action}")

        print("\nReview the intent plan and approve or reject it.")
        print("Enter 'approve' or 'reject [reason]'")

        timeout = 600  # 10 minutes
        start_time = time.time()

        while time.time() - start_time < timeout:
            try:
                response = input("Response: ").strip()

                if response.lower().startswith("approve"):
                    # Mark all intents as approved in the local copy
                    for intent in intents:
                        intent["status"] = "APPROVED"

                    intent_plan["status"] = "APPROVED"
                    intent_plan["review_timestamp"] = datetime.utcnow().isoformat()

                    return StringToolOutput(json.dumps({
                        "status": "approved",
                        "message": "Intent plan approved by reviewer",
                        "approved_plan": intent_plan # Return the modified plan
                    }, indent=2))

                elif response.lower().startswith("reject"):
                    reason = response[6:].strip() if len(response) > 6 else "No reason provided"

                    # Mark all intents as rejected in the local copy
                    for intent in intents:
                        intent["status"] = "REJECTED"

                    intent_plan["status"] = "REJECTED"
                    intent_plan["review_timestamp"] = datetime.utcnow().isoformat()
                    intent_plan["rejection_reason"] = reason

                    return StringToolOutput(json.dumps({
                        "status": "rejected",
                        "message": "Intent plan rejected by reviewer",
                        "reason": reason
                    }, indent=2))

                else:
                    print("Invalid response. Please enter 'approve' or 'reject [reason]'.")

            except KeyboardInterrupt:
                return StringToolOutput(json.dumps({
                    "status": "cancelled",
                    "message": "Review cancelled by user"
                }, indent=2))

            await asyncio.sleep(0.1)

        # Timeout
        return StringToolOutput(json.dumps({
            "status": "timeout",
            "message": f"Review timed out after {timeout} seconds"
        }, indent=2))