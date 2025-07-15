import unittest
from unittest.mock import MagicMock, AsyncMock, patch

from evolving_agents.core.intent_review import IntentReviewSystem, ReviewResult, ReviewStatus
# Tools used by IntentReviewAgent (or directly by the system)
from evolving_agents.tools.intent_review.component_selection_review_tool import ComponentSelectionReviewTool
from evolving_agents.tools.intent_review.workflow_design_review_tool import WorkflowDesignReviewTool

from evolving_agents.core.llm_service import LLMService
from evolving_agents.config import EvolvingAgentsConfig
# from beeai_framework.core.run_context import RunContext # For tools

# Placeholders
class PlaceholderRunContext(MagicMock):
    pass

class PlaceholderToolOutput(MagicMock):
    def __init__(self, content, **kwargs):
        super().__init__(**kwargs)
        self.content = content


try:
    from beeai_framework.core.run_context import RunContext
    # from beeai_framework.core.tool import ToolOutput # If tools return this
except ImportError:
    RunContext = PlaceholderRunContext
    # ToolOutput = PlaceholderToolOutput


class TestIntentReviewIntegration(unittest.IsolatedAsyncioTestCase):

    async def asyncSetUp(self):
        self.config = EvolvingAgentsConfig()
        self.mock_llm_service = MagicMock(spec=LLMService)
        # Mock LLMService.generate for the review tools
        self.mock_llm_service.generate = AsyncMock(return_value="LLM review: LGTM")

        # Instantiate review tools (they are called by IntentReviewSystem or its agent)
        self.component_review_tool = ComponentSelectionReviewTool(
            llm_service=self.mock_llm_service, config=self.config
        )
        self.workflow_review_tool = WorkflowDesignReviewTool(
            llm_service=self.mock_llm_service, config=self.config
        )

        # Mock the _run methods of the tools to control their output for tests
        # These tools' _run methods would typically call an LLM.
        self.component_review_tool._run = AsyncMock(
            return_value=PlaceholderToolOutput(content={
                "approved": True, "feedback": "Component selection looks good.", "confidence": 0.9
            })
        )
        self.workflow_review_tool._run = AsyncMock(
            return_value=PlaceholderToolOutput(content={
                "approved": True, "feedback": "Workflow design is sound.", "confidence": 0.85
            })
        )

        # IntentReviewSystem might use an agent (IntentReviewAgent) or call tools directly.
        # For this test, we'll assume IntentReviewSystem can use these tools.
        # If it uses an agent, we'd mock that agent's interaction with the tools.
        self.intent_review_system = IntentReviewSystem(
            llm_service=self.mock_llm_service, # Or pass tools directly if it uses them
            config=self.config
        )
        # Patch the tools if IntentReviewSystem instantiates them internally
        # For this example, let's assume it can be given tool instances, or we patch its _get_tool method
        self.intent_review_system.component_selection_tool = self.component_review_tool
        self.intent_review_system.workflow_design_tool = self.workflow_review_tool

        self.mock_run_context = MagicMock(spec=RunContext)


    async def test_review_plan_with_beeai_components_selection(self):
        """
        Test reviewing a plan that involves selecting BeeAI components.
        Ensures ComponentSelectionReviewTool is involved.
        """
        await self.asyncSetUp()

        plan_description = "Plan: Use BeeAI's ToolCallingAgent for customer service, with a SandboxTool."
        # This plan implies selection of 'ToolCallingAgent' and 'SandboxTool'
        # which are BeeAI components.

        # The IntentReviewSystem.review_intent method would determine which review tools to use.
        # We are interested in the call to ComponentSelectionReviewTool.

        review_result = await self.intent_review_system.review_intent(
            plan_description=plan_description,
            context=self.mock_run_context,
            # We might need to provide more structured plan details for the tools
            # For now, assume plan_description is enough for the mock tools to work.
            # In reality, specific components and workflow steps would be extracted and passed.
            components_to_review=[ # Simulating extracted components for the tool
                {"name": "CustomerServiceAgent", "type": "ToolCallingAgent", "provider": "BeeAIProvider"},
                {"name": "SecureCodeExecution", "type": "SandboxTool", "provider": "BeeAIProvider"}
            ]
        )

        # Check that ComponentSelectionReviewTool's _run was called
        self.component_review_tool._run.assert_called_once()
        call_args_kwargs = self.component_review_tool._run.call_args
        tool_input = call_args_kwargs[1]['input_data'] # input_data is a kwarg

        self.assertIn("ToolCallingAgent", tool_input.plan_details) # Or however the tool takes its input
        self.assertIn("SandboxTool", tool_input.plan_details)

        self.assertIsInstance(review_result, ReviewResult)
        self.assertEqual(review_result.status, ReviewStatus.APPROVED)
        self.assertIn("Component selection looks good.", review_result.feedback)


    async def test_review_plan_with_agentworkflow_design(self):
        """
        Test reviewing a plan that involves designing an AgentWorkflow (a BeeAI feature).
        Ensures WorkflowDesignReviewTool is involved.
        """
        await self.asyncSetUp()

        plan_description = "Plan: Implement an AgentWorkflow for invoice processing with steps: OCR, Extraction, Validation."
        # This implies designing a workflow using BeeAI's AgentWorkflow.

        # Mock the workflow tool to return approved
        self.workflow_review_tool._run.return_value = PlaceholderToolOutput(content={
            "approved": True, "feedback": "AgentWorkflow design is viable.", "confidence": 0.92
        })

        review_result = await self.intent_review_system.review_intent(
            plan_description=plan_description,
            context=self.mock_run_context,
            workflow_to_review={ # Simulating extracted workflow for the tool
                "name": "InvoiceProcessingWorkflow",
                "steps": [
                    {"agent": "OCR Agent", "role": "Extract text"},
                    {"agent": "Data Extractor Agent", "role": "Extract structured data"},
                    {"agent": "Validation Agent", "role": "Validate data"}
                ],
                "type": "AgentWorkflow" # Indicating BeeAI's AgentWorkflow
            }
        )

        # Check that WorkflowDesignReviewTool's _run was called
        self.workflow_review_tool._run.assert_called_once()
        call_args_kwargs = self.workflow_review_tool._run.call_args
        tool_input = call_args_kwargs[1]['input_data'] # input_data is a kwarg

        self.assertIn("AgentWorkflow", tool_input.plan_details) # Or however tool takes input
        self.assertEqual(len(tool_input.workflow_steps), 3)

        self.assertEqual(review_result.status, ReviewStatus.APPROVED)
        self.assertIn("AgentWorkflow design is viable.", review_result.feedback)

    # TODO:
    # - Test a scenario where review is REJECTED or NEEDS_DISCUSSION for BeeAI components.
    # - Test how IntentReviewSystem combines feedback from multiple review tools if applicable.
    # - Test if IntentReviewAgent (if used by the system) correctly uses these tools.

if __name__ == '__main__':
    unittest.main()
