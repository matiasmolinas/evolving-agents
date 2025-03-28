# examples/invoice_processing/architect_zero_comprehensive_demo_refactored.py

import asyncio
import logging
import json
import os
import shutil
from typing import Dict, Any, Optional, List # Added List
from dotenv import load_dotenv
import re # Import re for extraction

# --- Evolving Agents Imports ---
from evolving_agents.core.llm_service import LLMService
from evolving_agents.smart_library.smart_library import SmartLibrary
from evolving_agents.agent_bus.smart_agent_bus import SmartAgentBus
from evolving_agents.core.system_agent import SystemAgentFactory
from evolving_agents.agents.architect_zero import create_architect_zero
from evolving_agents.core.dependency_container import DependencyContainer
from evolving_agents.firmware.firmware import Firmware

# --- Configuration ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
load_dotenv() # Load API keys etc. from .env file

# --- Constants ---
SMART_LIBRARY_PATH = "smart_library_demo.json"
AGENT_BUS_PATH = "smart_agent_bus_demo.json"
AGENT_BUS_LOG_PATH = "agent_bus_logs_demo.json"
VECTOR_DB_PATH = "./vector_db_demo"
CACHE_DIR = ".llm_cache_demo"
ARCHITECT_DESIGN_PATH = "architect_design_output.json"
ARCHITECT_RAW_OUTPUT_PATH = "architect_raw_output.txt" # For saving raw output if JSON fails
GENERATED_WORKFLOW_PATH = "generated_invoice_workflow.yaml"
WORKFLOW_OUTPUT_PATH = "workflow_execution_output.json"
SAMPLE_INVOICE_PATH = "sample_invoice_input.txt"

# Sample invoice for testing
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
        SMART_LIBRARY_PATH,
        AGENT_BUS_PATH,
        AGENT_BUS_LOG_PATH,
        VECTOR_DB_PATH,
        CACHE_DIR,
        ARCHITECT_DESIGN_PATH,
        ARCHITECT_RAW_OUTPUT_PATH, # Added raw output path
        GENERATED_WORKFLOW_PATH,
        WORKFLOW_OUTPUT_PATH,
        SAMPLE_INVOICE_PATH
    ]
    for item_path in items_to_remove:
        try:
            if os.path.exists(item_path):
                if os.path.isdir(item_path):
                    shutil.rmtree(item_path)
                    logger.info(f"Removed directory: {item_path}")
                else:
                    os.remove(item_path)
                    logger.info(f"Removed file: {item_path}")
        except Exception as e:
            logger.warning(f"Could not remove {item_path}: {e}")
    logger.info("Cleanup complete.")

async def setup_initial_library(library_path: str, llm_service: LLMService):
    """Seed the library with basic components for the demo."""
    logger.info("--- Setting up initial Smart Library components ---")
    # Pass vector_db_path correctly
    smart_library = SmartLibrary(library_path, llm_service=llm_service, vector_db_path=VECTOR_DB_PATH)

    # 1. Basic Document Analyzer Tool (BeeAI)
    # Using a simplified but functional code snippet for brevity in the demo setup
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
        name="BasicDocumentAnalyzer",
        record_type="TOOL", domain="document_processing",
        description="A basic tool that analyzes documents to guess their type (invoice, receipt, etc.) using keyword matching.",
        code_snippet=basic_doc_analyzer_code,
        version="1.0", tags=["document", "analysis", "basic", "beeai"],
        metadata={"framework": "beeai"}
    )
    logger.info("Created: BasicDocumentAnalyzer (Tool)")

    # 2. Basic Invoice Processor Agent (BeeAI)
    # Simplified initializer for demo setup
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
    # Example method (not directly used by ReAct Agent, but illustrative of capability)
    @staticmethod
    def extract_info(invoice_text: str) -> Dict[str, Any]:
        # Simple extraction logic...
        num = re.search(r'INVOICE #([\\w-]+)', invoice_text, re.I); dte = re.search(r'Date:?\\s*([\\w\\d/-]+)', invoice_text, re.I)
        vnd = re.search(r'Vendor:?\\s*([^\\n]+)', invoice_text, re.I); tot = re.search(r'Total\\s*(?:Due|Amount)?:?\\s*\\$?([\\d.,]+)', invoice_text, re.I)
        return { "invoice_number": num.group(1) if num else '?', "date": dte.group(1).strip() if dte else '?', "vendor": vnd.group(1).strip() if vnd else '?', "total": tot.group(1).strip() if tot else '?'}
"""
    await smart_library.create_record(
        name="BasicInvoiceProcessor",
        record_type="AGENT", domain="financial",
        description="A basic BeeAI agent that extracts invoice number, date, vendor, and total using simple regex.",
        code_snippet=basic_invoice_processor_code,
        version="1.0", tags=["invoice", "processing", "basic", "beeai", "finance"],
        metadata={"framework": "beeai"}
    )
    logger.info("Created: BasicInvoiceProcessor (Agent)")

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
        name="WeatherForecaster",
        record_type="TOOL", domain="utilities",
        description="Provides mock weather forecasts.",
        code_snippet=weather_tool_code,
        version="1.0", tags=["weather", "mock", "beeai"],
        metadata={"framework": "beeai"}
    )
    logger.info("Created: WeatherForecaster (Tool)")

    # Allow vector DB sync
    logger.info("Waiting for initial vector DB sync...")
    await asyncio.sleep(4) # Slightly longer wait
    logger.info("Initial library setup complete.")
    return smart_library

def extract_json_from_response(response_text: str) -> Optional[Dict[str, Any]]:
    """Tries various methods to extract a JSON object from LLM response text."""
    # 1. Try direct parsing (if the whole string is JSON)
    try:
        # Clean potential leading/trailing whitespace or markdown markers
        cleaned_text = response_text.strip()
        if cleaned_text.startswith("```json"):
            cleaned_text = cleaned_text[len("```json"):].strip()
        if cleaned_text.startswith("```"):
             cleaned_text = cleaned_text[len("```"):].strip()
        if cleaned_text.endswith("```"):
            cleaned_text = cleaned_text[:-len("```")].strip()

        if cleaned_text.startswith('{') and cleaned_text.endswith('}'):
            return json.loads(cleaned_text)
    except json.JSONDecodeError:
        pass # Continue to next method

    # 2. Look for JSON within ```json ... ``` markdown blocks explicitly
    json_match = re.search(r"```json\s*([\s\S]*?)\s*```", response_text, re.MULTILINE)
    if json_match:
        try:
            return json.loads(json_match.group(1))
        except json.JSONDecodeError:
            logger.warning("Found JSON markdown block, but content was invalid JSON.")
            pass # Continue

    # 3. Look for JSON within generic ``` ... ``` markdown blocks
    code_match = re.search(r"```\s*([\s\S]*?)\s*```", response_text, re.MULTILINE)
    if code_match:
        try:
            potential_json = code_match.group(1).strip()
            if potential_json.startswith('{') and potential_json.endswith('}'):
                 return json.loads(potential_json)
        except json.JSONDecodeError:
            logger.warning("Found generic markdown block, but content was not valid JSON.")
            pass # Continue

    # 4. Look for the first '{' and last '}' as a last resort
    start_index = response_text.find('{')
    end_index = response_text.rfind('}')
    if start_index != -1 and end_index != -1 and end_index > start_index:
        potential_json = response_text[start_index : end_index + 1]
        try:
            # Basic validation: ensure roughly balanced braces
            if potential_json.count('{') == potential_json.count('}'):
                 return json.loads(potential_json)
        except json.JSONDecodeError:
            logger.warning("Could not parse content between first '{' and last '}'.")
            pass # Failed

    logger.error("Failed to extract valid JSON using multiple methods.")
    return None

def extract_yaml_from_response(response_text: str) -> Optional[str]:
    """Extracts the first YAML block from the LLM's response."""
    # 1. Look for ```yaml ... ``` markdown blocks explicitly
    match = re.search(r"```yaml\s*([\s\S]*?)\s*```", response_text, re.MULTILINE)
    if match:
        return match.group(1).strip()

    # 2. Fallback: Look for text starting like YAML if no markers found
    lines = response_text.strip().split('\n')
    if lines and (lines[0].strip().startswith("scenario_name:") or lines[0].strip().startswith("- type:")):
        yaml_lines = []
        for line in lines:
            # Basic heuristic: stop if a line looks like explanatory text
            if not line.startswith(' ') and ':' not in line and not line.strip().startswith('-') and len(yaml_lines) > 1:
                 # Allow simple comments starting with #
                 if not line.strip().startswith('#'):
                      break
            yaml_lines.append(line)
        if yaml_lines:
            return "\n".join(yaml_lines).strip()

    logger.warning("Could not reliably extract YAML from response.")
    return None

# --- Main Demo Function ---

async def run_demo():
    """Runs the comprehensive ArchitectZero and SystemAgent demo."""
    logger.info("=================================================")
    logger.info("=    Evolving Agents - Comprehensive Demo     =")
    logger.info("=================================================")

    # --- Phase 0: Setup ---
    clean_previous_state()
    with open(SAMPLE_INVOICE_PATH, "w") as f:
        f.write(SAMPLE_INVOICE)
    logger.info(f"Saved sample invoice to {SAMPLE_INVOICE_PATH}")

    container = DependencyContainer()

    # --- Phase 1: Initializing Core Services ---
    logger.info("--- Phase 1: Initializing Core Services ---")
    llm_service = LLMService(provider="openai", model="gpt-4o-mini", cache_dir=CACHE_DIR)
    container.register('llm_service', llm_service)
    logger.info(f"LLM Service initialized (Provider: {llm_service.provider}, Model: {llm_service.chat_model.model_id})")

    smart_library = await setup_initial_library(SMART_LIBRARY_PATH, llm_service)
    container.register('smart_library', smart_library)
    logger.info(f"Smart Library initialized ({len(smart_library.records)} initial records)")

    firmware = Firmware()
    container.register('firmware', firmware)
    logger.info("Firmware component initialized")

    agent_bus = SmartAgentBus(
        storage_path=AGENT_BUS_PATH, log_path=AGENT_BUS_LOG_PATH,
        smart_library=smart_library, llm_service=llm_service, container=container,
        chroma_path=VECTOR_DB_PATH # Pass vector path
    )
    container.register('agent_bus', agent_bus)
    logger.info("Smart Agent Bus initialized")

    # --- Phase 2: Initializing Agents ---
    logger.info("--- Phase 2: Initializing Agents ---")
    system_agent = await SystemAgentFactory.create_agent(container=container)
    container.register('system_agent', system_agent)
    logger.info("System Agent initialized with its tools")

    architect_agent = await create_architect_zero(container=container)
    container.register('architect_agent', architect_agent)
    logger.info("ArchitectZero Agent initialized")

    # Finalize initializations
    agent_bus.system_agent = system_agent
    # Library already initialized in setup_initial_library
    # await smart_library.initialize() # Not needed again
    await agent_bus.initialize_from_library() # Register library components with the bus
    logger.info("Core services and agents fully initialized and connected.")


    # --- Phase 3: ArchitectZero Designs the Solution ---
    logger.info("--- Phase 3: ArchitectZero Designing the Solution ---")
    task_requirement = """
    Design an advanced invoice processing system. The current library has basic components ('BasicDocumentAnalyzer', 'BasicInvoiceProcessor').
    Your design should improve upon them significantly:

    1.  **Component Analysis:** Analyze existing components relevant to 'document_processing' and 'financial' domains. Note their limitations (e.g., BasicDocumentAnalyzer uses simple keywords, BasicInvoiceProcessor uses basic regex).
    2.  **Enhanced Analyzer:** Design a *new* tool named 'AdvancedDocumentAnalyzer'. It should use heuristics (presence of keywords like 'invoice', 'bill', date patterns, monetary values '$X.XX', total/subtotal indicators, vendor info, line items) to calculate a confidence score for identifying invoices. Aim for >0.8 confidence for a positive ID. Specify its input (text) and output (JSON with 'type' and 'confidence'). Action: 'create'.
    3.  **Evolved Processor:** Design an *evolved* version of 'BasicInvoiceProcessor' named 'VerifiedInvoiceProcessor'. This *agent* should:
        *   Extract comprehensive fields: Invoice Number, Date, Vendor Name, Bill To details, Line Items (Description, Quantity, Unit Price, Item Total), Subtotal, Tax Amount, Shipping (if present), Total Due, Payment Terms, Due Date.
        *   Perform calculation verification: Sum of Line Item Totals should match Subtotal; Subtotal + Tax + Shipping (if present) should match Total Due.
        *   Output results as a structured JSON object. Include a 'verification' field containing 'status' ('ok' or 'failed') and a list of 'discrepancies'. Action: 'evolve', existing_component_id: 'BasicInvoiceProcessor' (or its actual ID if known).
    4.  **Workflow Design:** Define the sequence of operations:
        a. Input: Raw invoice text.
        b. Step 1: Use 'AdvancedDocumentAnalyzer' on the input text. Store output in 'analysis_result'.
        c. Step 2: Condition: If 'analysis_result.type' is 'invoice' AND 'analysis_result.confidence' > 0.8.
        d. Step 2a (if condition met): Use 'VerifiedInvoiceProcessor' with the original invoice text. Store output in 'extracted_data'.
        e. Step 3: Return 'extracted_data' (or an appropriate message if not an invoice).
    5.  **Output:** Produce ONLY a valid JSON object representing the complete solution design (containing keys like "overview", "architecture", "components", "workflow"). Do not include any text before or after the JSON object itself.
    """
    logger.info(f"Giving task to ArchitectZero:\n{task_requirement[:400]}...") # Log slightly more

    solution_design = None # Initialize to None
    try:
        architect_response = await architect_agent.run(task_requirement)
        design_output_text = architect_response.result.text if architect_response and hasattr(architect_response.result, 'text') else str(architect_response)
        logger.info("ArchitectZero finished designing.")

        solution_design = extract_json_from_response(design_output_text)

        if solution_design:
            with open(ARCHITECT_DESIGN_PATH, "w") as f:
                json.dump(solution_design, f, indent=2)
            logger.info(f"Architect's solution design (JSON) extracted and saved to {ARCHITECT_DESIGN_PATH}")
            # print("\n--- Solution Design ---")
            # print(json.dumps(solution_design, indent=2))
            # print("---------------------\n")
        else:
            logger.warning("Architect's output could not be parsed as valid JSON.")
            with open(ARCHITECT_RAW_OUTPUT_PATH, "w") as f:
                f.write(design_output_text)
            logger.info(f"Architect's raw output saved to {ARCHITECT_RAW_OUTPUT_PATH}")

    except Exception as e:
        logger.error(f"Error running ArchitectZero or processing its output: {e}", exc_info=True)
        solution_design = None

    # --- Phase 4: SystemAgent Generates the Workflow ---
    logger.info("--- Phase 4: SystemAgent Generating Executable Workflow ---")
    generated_yaml_workflow = None
    if solution_design: # Proceed only if we have a valid design
        # Construct prompt for SystemAgent using the architect's design
        workflow_generation_prompt = f"""
        Based on the following solution design JSON produced by ArchitectZero, please use your 'GenerateWorkflowTool'
        to create an executable YAML workflow.

        Solution Design:
        ```json
        {json.dumps(solution_design, indent=2)}
        ```

        Ensure the YAML includes:
        1. A `DEFINE` step for the *new* 'AdvancedDocumentAnalyzer' tool, including its `code_snippet` (generate plausible BeeAI tool code based on the design's description).
        2. A `DEFINE` step for the *evolved* 'VerifiedInvoiceProcessor' agent, referencing 'BasicInvoiceProcessor' via `from_existing_snippet` and including its `code_snippet` (generate plausible BeeAI agent code based on the design's description and evolution needs).
        3. `CREATE` steps for both 'AdvancedDocumentAnalyzer' and 'VerifiedInvoiceProcessor'.
        4. `EXECUTE` steps following the workflow sequence designed by the architect, using `output_var` to pass data and `condition` for branching. Use `{{{{params.invoice_text}}}}` for the initial input.
        5. A final `RETURN` step for the extracted data.

        Return *only* the generated YAML content, enclosed in ```yaml ... ``` blocks.
        """
        logger.info("Asking SystemAgent to generate YAML workflow from the design...")

        try:
            sys_agent_gen_response = await system_agent.run(workflow_generation_prompt)
            sys_agent_gen_output = sys_agent_gen_response.result.text if sys_agent_gen_response and hasattr(sys_agent_gen_response.result, 'text') else str(sys_agent_gen_response)

            # Extract the YAML from the SystemAgent's response
            extracted_yaml = extract_yaml_from_response(sys_agent_gen_output)

            if extracted_yaml:
                generated_yaml_workflow = extracted_yaml
                with open(GENERATED_WORKFLOW_PATH, "w") as f:
                    f.write(generated_yaml_workflow)
                logger.info(f"SystemAgent generated workflow YAML saved to {GENERATED_WORKFLOW_PATH}")
                # Optional: Print generated YAML for debugging
                # print("\n--- Generated YAML ---")
                # print(generated_yaml_workflow)
                # print("--------------------\n")
            else:
                logger.warning("SystemAgent did not return a valid YAML workflow in its response.")
                logger.debug(f"SystemAgent raw response for generation:\n{sys_agent_gen_output}")

        except Exception as e:
            logger.error(f"Error asking SystemAgent to generate workflow: {e}", exc_info=True)
    else:
        logger.warning("Skipping workflow generation as ArchitectZero did not produce a valid JSON design.")


    # --- Phase 5: SystemAgent Executes the Workflow ---
    logger.info("--- Phase 5: SystemAgent Executing the Workflow ---")
    if generated_yaml_workflow: # Proceed only if we have YAML
        workflow_execution_prompt = f"""
        Please execute the following workflow.
        Use your 'ProcessWorkflowTool' to parse the YAML and get the execution plan.
        Then, execute the steps in the plan sequentially using your other tools (CreateComponentTool, EvolveComponentTool, etc.).
        Substitute the parameters provided.

        Workflow YAML:
        ```yaml
        {generated_yaml_workflow}
        ```

        Parameters for this execution:
        {{
            "invoice_text": {json.dumps(SAMPLE_INVOICE)}
        }}

        Report the final outcome after executing all steps. The workflow's RETURN step specifies the final value. Please provide this final value, ideally as a JSON object.
        """
        logger.info("Asking SystemAgent to execute the generated workflow...")

        try:
            sys_agent_exec_response = await system_agent.run(workflow_execution_prompt)
            execution_output_text = sys_agent_exec_response.result.text if sys_agent_exec_response and hasattr(sys_agent_exec_response.result, 'text') else str(sys_agent_exec_response)

            # Try to extract the final JSON result from the agent's *overall* output
            final_result_json = extract_json_from_response(execution_output_text)
            if not final_result_json:
                 # If no JSON found in overall output, maybe the agent just output the JSON directly?
                 # Let's re-check the very end of the text
                 potential_json_at_end = execution_output_text.split('\n')[-1].strip()
                 if potential_json_at_end.startswith('{') and potential_json_at_end.endswith('}'):
                     try:
                         final_result_json = json.loads(potential_json_at_end)
                     except json.JSONDecodeError:
                         final_result_json = None # Still couldn't parse

            if not final_result_json:
                 logger.warning("Could not parse final JSON result from SystemAgent execution output. Using raw text.")
                 final_result_data = execution_output_text
            else:
                 final_result_data = final_result_json


            execution_result_structured = {
                "status": "completed",
                "final_result": final_result_data,
                "agent_full_output": execution_output_text # Keep the full log for debugging
            }

            with open(WORKFLOW_OUTPUT_PATH, "w") as f:
                json.dump(execution_result_structured, f, indent=2)
            logger.info(f"Workflow execution finished. Results saved to {WORKFLOW_OUTPUT_PATH}")

            logger.info("\n--- Workflow Execution Final Result ---")
            if isinstance(final_result_data, dict):
                logger.info(json.dumps(final_result_data, indent=2))
            else:
                 logger.info(final_result_data) # Print raw text if not JSON
            logger.info("-------------------------------------\n")

        except Exception as e:
            logger.error(f"Error asking SystemAgent to execute workflow: {e}", exc_info=True)
    else:
        logger.warning("Skipping workflow execution as no valid YAML workflow was generated.")

    logger.info("=================================================")
    logger.info("=             Demo Run Finished                 =")
    logger.info("=================================================")
    logger.info(f"Check generated files: {ARCHITECT_DESIGN_PATH}, {GENERATED_WORKFLOW_PATH}, {WORKFLOW_OUTPUT_PATH}")
    if not solution_design:
         logger.warning(f"Architect design failed JSON parsing. Raw output saved to: {ARCHITECT_RAW_OUTPUT_PATH}")

# --- Run the Demo ---
if __name__ == "__main__":
    try:
        asyncio.run(run_demo())
    except Exception as main_error:
        logger.critical(f"An unhandled error occurred in the main demo loop: {main_error}", exc_info=True)