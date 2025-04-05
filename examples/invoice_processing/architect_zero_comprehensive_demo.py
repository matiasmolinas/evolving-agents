# examples/invoice_processing/architect_zero_comprehensive_demo.py

import asyncio
import logging
import json
import os
import shutil
from typing import Dict, Any, Optional, List
from dotenv import load_dotenv
import re

# --- Evolving Agents Imports ---
from evolving_agents.core.llm_service import LLMService
from evolving_agents.smart_library.smart_library import SmartLibrary
from evolving_agents.agent_bus.smart_agent_bus import SmartAgentBus
from evolving_agents.core.system_agent import SystemAgentFactory
# NOTE: We no longer import create_architect_zero directly into the demo script
from evolving_agents.core.dependency_container import DependencyContainer
from evolving_agents.firmware.firmware import Firmware
# Import for type checking during setup (optional)
from evolving_agents.agents.architect_zero import ArchitectZeroAgentInitializer

# --- Configuration ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
load_dotenv()

# --- Constants ---
SMART_LIBRARY_PATH = "smart_library_demo.json"
AGENT_BUS_PATH = "smart_agent_bus_demo.json"
AGENT_BUS_LOG_PATH = "agent_bus_logs_demo.json"
VECTOR_DB_PATH = "./vector_db_demo"
CACHE_DIR = ".llm_cache_demo"
# ARCHITECT_DESIGN_PATH = "architect_design_output.json" # No longer explicitly managed by demo
# GENERATED_WORKFLOW_PATH = "generated_invoice_workflow.yaml" # No longer explicitly managed by demo
WORKFLOW_OUTPUT_PATH = "final_processing_output.json" # Renamed for clarity
SAMPLE_INVOICE_PATH = "sample_invoice_input.txt"
SAMPLE_INVOICE = """
INVOICE #INV-9876
Date: 2024-07-15
Vendor: Advanced Tech Solutions
Address: 555 Innovation Drive, Tech Park, CA 95054

Bill To:
Global Corp Inc.
1 Enterprise Plaza
Metropolis, NY 10020

Items:
1. Quantum Entanglement Router - $15,000.00 (1 unit)
2. AI Co-processor Module    - $2,500.00 (4 units)
3. Holo-Display Unit        - $1,200.00 (2 units)

Subtotal: $27,400.00
Tax (9.25%): $2,534.50
Shipping: $150.00
Total Due: $30,084.50

Payment Terms: Net 45
Due Date: 2024-08-29

Notes: Expedited shipping requested.
"""

# --- Helper Functions ---
def clean_previous_state():
    """Remove previous demo files and directories to start fresh."""
    logger.info("--- Cleaning up previous demo state ---")
    items_to_remove = [
        SMART_LIBRARY_PATH, AGENT_BUS_PATH, AGENT_BUS_LOG_PATH,
        VECTOR_DB_PATH, CACHE_DIR, WORKFLOW_OUTPUT_PATH, SAMPLE_INVOICE_PATH,
        # Remove files no longer explicitly created by demo:
        "architect_design_output.json", "architect_raw_output.txt",
        "generated_invoice_workflow.yaml"
    ]
    for item_path in items_to_remove:
        try:
            if os.path.exists(item_path):
                if os.path.isdir(item_path): shutil.rmtree(item_path)
                else: os.remove(item_path)
                logger.info(f"Removed: {item_path}")
        except Exception as e: logger.warning(f"Could not remove {item_path}: {e}")
    logger.info("Cleanup complete.")

async def setup_framework_environment(container: DependencyContainer):
    """Initializes all core framework services and agents."""
    logger.info("--- Initializing Framework Environment ---")

    # 1. LLM Service
    llm_service = LLMService(provider="openai", model="gpt-4o-mini", cache_dir=CACHE_DIR)
    container.register('llm_service', llm_service)
    logger.info(f"LLM Service initialized (Provider: {llm_service.provider}, Model: {llm_service.chat_model.model_id})")

    # 2. Smart Library (with initial seeding)
    smart_library = SmartLibrary(SMART_LIBRARY_PATH, llm_service=llm_service, vector_db_path=VECTOR_DB_PATH, container=container)
    container.register('smart_library', smart_library)
    await seed_initial_library(smart_library) # Seed with basic components
    logger.info(f"Smart Library initialized ({len(smart_library.records)} initial records)")

    # 3. Firmware
    firmware = Firmware()
    container.register('firmware', firmware)
    logger.info("Firmware component initialized")

    # 4. Agent Bus
    agent_bus = SmartAgentBus(container=container, storage_path=AGENT_BUS_PATH, log_path=AGENT_BUS_LOG_PATH, chroma_path=VECTOR_DB_PATH)
    container.register('agent_bus', agent_bus)
    logger.info("Smart Agent Bus initialized")

    # 5. System Agent (Depends on others, created via factory)
    system_agent = await SystemAgentFactory.create_agent(container=container)
    container.register('system_agent', system_agent)
    logger.info("System Agent initialized")

    # 6. Architect Agent (Initialized but not called directly by demo)
    # We ensure it's created so SystemAgent *could* potentially call it via the bus if needed
    architect_agent = await ArchitectZeroAgentInitializer.create_agent(container=container)
    container.register('architect_agent', architect_agent)
    logger.info("ArchitectZero Agent initialized (available on bus)")

    # 7. Finalize Bus Initialization (Connects agents registered from library)
    await agent_bus.initialize_from_library()
    logger.info("Framework Environment fully initialized and connected.")
    return container # Return container with all initialized components

async def seed_initial_library(smart_library: SmartLibrary):
    """Seed the library with basic components needed for context."""
    logger.info("--- Seeding initial Smart Library components ---")
    # (Keep the record creation logic from the previous script)
    # 1. Basic Document Analyzer Tool (BeeAI)
    basic_doc_analyzer_code = """
from typing import Dict; import json; from pydantic import BaseModel, Field
from beeai_framework.tools.tool import StringToolOutput, Tool
from beeai_framework.context import RunContext; from beeai_framework.emitter import Emitter
class Input(BaseModel): text: str
class BasicDocumentAnalyzer(Tool[Input, None, StringToolOutput]):
    name = "BasicDocumentAnalyzer"; description = "Guesses doc type based on keywords"
    input_schema = Input
    def _create_emitter(self) -> Emitter: return Emitter.root().child(creator=self)
    async def _run(self, input: Input, options=None, context=None) -> StringToolOutput:
        txt = input.text.lower(); r = {"type":"unknown","confidence":0.0}
        if "invoice" in txt and ("total" in txt or "amount due" in txt): r = {"type":"invoice","confidence":0.7}
        elif "receipt" in txt: r = {"type":"receipt","confidence":0.6}
        elif "contract" in txt: r = {"type":"contract","confidence":0.6}
        return StringToolOutput(json.dumps(r))
"""
    await smart_library.create_record(
        name="BasicDocumentAnalyzer", record_type="TOOL", domain="document_processing",
        description="A basic tool that analyzes documents to guess their type (invoice, receipt, etc.) using keyword matching.",
        code_snippet=basic_doc_analyzer_code, version="1.0", tags=["document", "analysis", "basic", "beeai"],
        metadata={"framework": "beeai"}
    )
    logger.info("Seeded: BasicDocumentAnalyzer (Tool)")
    # 2. Basic Invoice Processor Agent (BeeAI)
    basic_invoice_processor_code = """
from typing import List, Dict, Any, Optional; import re; import json
from beeai_framework.agents.react import ReActAgent; from beeai_framework.agents.types import AgentMeta
from beeai_framework.memory import TokenMemory; from beeai_framework.backend.chat import ChatModel
from beeai_framework.tools.tool import Tool
class BasicInvoiceProcessorInitializer:
    '''Basic agent to extract invoice number, date, vendor, total.'''
    @staticmethod
    def create_agent(llm: ChatModel, tools: Optional[List[Tool]] = None) -> ReActAgent:
        meta = AgentMeta(name="BasicInvoiceProcessor", description="I extract basic invoice info: number, date, vendor, total.", tools=tools or [])
        agent = ReActAgent(llm=llm, tools=tools or [], memory=TokenMemory(llm), meta=meta)
        return agent
    @staticmethod
    def extract_info(invoice_text: str) -> Dict[str, Any]:
        num = re.search(r'INVOICE #([\\w-]+)', invoice_text, re.I); dte = re.search(r'Date:?\\s*([\\w\\d/-]+)', invoice_text, re.I)
        vnd = re.search(r'Vendor:?\\s*([^\\n]+)', invoice_text, re.I); tot = re.search(r'Total\\s*(?:Due|Amount)?:?\\s*\\$?([\\d.,]+)', invoice_text, re.I)
        return { "invoice_number": num.group(1) if num else '?', "date": dte.group(1).strip() if dte else '?', "vendor": vnd.group(1).strip() if vnd else '?', "total": tot.group(1).strip() if tot else '?'}
"""
    await smart_library.create_record(
        name="BasicInvoiceProcessor", record_type="AGENT", domain="financial",
        description="A basic BeeAI agent that extracts invoice number, date, vendor, and total using simple regex.",
        code_snippet=basic_invoice_processor_code, version="1.0", tags=["invoice", "processing", "basic", "beeai", "finance"],
        metadata={"framework": "beeai"}
    )
    logger.info("Seeded: BasicInvoiceProcessor (Agent)")
    # 3. Add a completely unrelated tool for distraction
    weather_tool_code = """
from pydantic import BaseModel, Field; from beeai_framework.tools.tool import Tool, StringToolOutput
from beeai_framework.context import RunContext; from beeai_framework.emitter import Emitter
class Input(BaseModel): location: str
class WeatherForecaster(Tool[Input, None, StringToolOutput]):
    name="WeatherForecaster"; description="Gives fake weather forecast"; input_schema=Input
    def _create_emitter(self) -> Emitter: return Emitter.root().child(creator=self)
    async def _run(self, input: Input, options=None, context=None) -> StringToolOutput:
        return StringToolOutput(f"Weather for {input.location}: Sunny, 25C")
"""
    await smart_library.create_record(
        name="WeatherForecaster", record_type="TOOL", domain="utilities",
        description="Provides mock weather forecasts.", code_snippet=weather_tool_code,
        version="1.0", tags=["weather", "mock", "beeai"], metadata={"framework": "beeai"}
    )
    logger.info("Seeded: WeatherForecaster (Tool)")
    logger.info("Waiting for initial vector DB sync after seeding...")
    await asyncio.sleep(5) # Allow time for embedding/indexing
    logger.info("Library seeding complete.")


def extract_json_from_response(response_text: str) -> Optional[Dict[str, Any]]:
    """Tries various methods to extract a JSON object from LLM response text."""
    # (Keep the robust JSON extraction logic)
    # 1. Try direct parsing
    try:
        cleaned_text = response_text.strip()
        if cleaned_text.startswith("```json"): cleaned_text = cleaned_text[len("```json"):].strip()
        if cleaned_text.startswith("```"): cleaned_text = cleaned_text[len("```"):].strip()
        if cleaned_text.endswith("```"): cleaned_text = cleaned_text[:-len("```")].strip()
        if cleaned_text.startswith('{') and cleaned_text.endswith('}'): return json.loads(cleaned_text)
    except json.JSONDecodeError: pass
    # 2. Look for ```json ... ```
    json_match = re.search(r"```json\s*([\s\S]*?)\s*```", response_text, re.MULTILINE)
    if json_match:
        try: return json.loads(json_match.group(1))
        except json.JSONDecodeError: logger.warning("Found JSON markdown block, but content was invalid JSON.")
    # 3. Look for ``` ... ```
    code_match = re.search(r"```\s*([\s\S]*?)\s*```", response_text, re.MULTILINE)
    if code_match:
        try:
            potential_json = code_match.group(1).strip()
            if potential_json.startswith('{') and potential_json.endswith('}'): return json.loads(potential_json)
        except json.JSONDecodeError: logger.warning("Found generic markdown block, but content was not valid JSON.")
    # 4. Look for first '{' and last '}'
    start_index = response_text.find('{'); end_index = response_text.rfind('}')
    if start_index != -1 and end_index != -1 and end_index > start_index:
        potential_json = response_text[start_index : end_index + 1]
        try:
            if potential_json.count('{') == potential_json.count('}'): return json.loads(potential_json)
        except json.JSONDecodeError: logger.warning("Could not parse content between first '{' and last '}'.")
    logger.warning("Failed to extract valid JSON from final response using multiple methods.")
    return None

# --- Main Demo Function ---

async def run_demo():
    """Runs the comprehensive demo focusing on a high-level task for SystemAgent."""
    logger.info("=================================================")
    logger.info("=    Evolving Agents - High-Level Task Demo    =")
    logger.info("=================================================")

    # --- Phase 0: Setup ---
    clean_previous_state()
    with open(SAMPLE_INVOICE_PATH, "w") as f: f.write(SAMPLE_INVOICE)
    logger.info(f"Saved sample invoice to {SAMPLE_INVOICE_PATH}")
    container = DependencyContainer()

    # --- Phase 1: Initialize Framework ---
    # This now encapsulates all setup including agents
    container = await setup_framework_environment(container)
    system_agent = container.get('system_agent') # Get the initialized SystemAgent

    # --- Phase 2: Define High-Level Task for SystemAgent ---
    logger.info("--- Phase 2: Defining High-Level Task for SystemAgent ---")

    # The prompt now focuses *only* on the functional/non-functional requirements
    # It does NOT mention ArchitectZero, designs, YAML, or specific tools.
    high_level_task_prompt = f"""
    **Goal:** Accurately process the provided invoice document and return structured, verified data.

    **Functional Requirements:**
    - Extract key fields: Invoice #, Date, Vendor, Bill To, Line Items (Description, Quantity, Unit Price, Item Total), Subtotal, Tax Amount, Shipping (if present), Total Due, Payment Terms, Due Date.
    - Verify calculations: The sum of line item totals should match the Subtotal. The sum of Subtotal, Tax Amount, and Shipping (if present) must match the Total Due. Report any discrepancies.

    **Non-Functional Requirements:**
    - High accuracy is critical.
    - Output must be a single, valid JSON object containing the extracted data and a 'verification' section (status: 'ok'/'failed', discrepancies: list).
    - Handle potential variations in invoice layouts gracefully.

    **Input Data:**
    ```
    {SAMPLE_INVOICE}
    ```

    **Action:** Achieve this goal using the best approach available within the framework. Create, evolve, or reuse components as needed. Return ONLY the final JSON result.
    """
    logger.info("Giving high-level task to SystemAgent...")


    # --- Phase 3: SystemAgent Executes the Task ---
    logger.info("--- Phase 3: SystemAgent Executing the High-Level Task ---")

    try:
        # Make the SINGLE call to the SystemAgent with the high-level task
        sys_agent_response = await system_agent.run(high_level_task_prompt)
        final_output_text = sys_agent_response.result.text if sys_agent_response and hasattr(sys_agent_response.result, 'text') else str(sys_agent_response)

        # --- Post-processing the output ---
        final_result_json = extract_json_from_response(final_output_text)

        if final_result_json:
             final_result_data = final_result_json
             logger.info("Successfully extracted final JSON result from SystemAgent output.")
        else:
             logger.warning("Could not parse final JSON result from SystemAgent execution output. Using raw text.")
             final_result_data = final_output_text # Use the raw text if no JSON found

        # Save the structured result including the agent's full response
        execution_result_structured = {
            "status": "completed" if final_result_json else "completed_with_warnings",
            "final_result": final_result_data,
            "agent_full_output": final_output_text # Keep the full log for debugging
        }
        with open(WORKFLOW_OUTPUT_PATH, "w") as f:
            json.dump(execution_result_structured, f, indent=2)
        logger.info(f"SystemAgent task execution finished. Results saved to {WORKFLOW_OUTPUT_PATH}")

        logger.info("\n--- System Agent Task Final Result ---")
        if isinstance(final_result_data, dict):
            logger.info(json.dumps(final_result_data, indent=2))
        else:
             logger.info(final_result_data) # Print raw text if not JSON
        logger.info("-------------------------------------\n")

    except Exception as e:
        logger.error(f"Error asking SystemAgent to execute the task: {e}", exc_info=True)

    logger.info("=================================================")
    logger.info("=             Demo Run Finished                 =")
    logger.info("=================================================")
    logger.info(f"Check final output file: {WORKFLOW_OUTPUT_PATH}")
    logger.info(f"(Intermediate files like design/workflow YAML are managed internally by the agent)")

# --- Run the Demo ---
if __name__ == "__main__":
    try:
        asyncio.run(run_demo())
    except Exception as main_error:
        logger.critical(f"An unhandled error occurred in the main demo loop: {main_error}", exc_info=True)