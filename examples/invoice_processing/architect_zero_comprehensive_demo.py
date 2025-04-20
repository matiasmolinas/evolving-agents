# examples/invoice_processing/architect_zero_comprehensive_demo.py

import asyncio
import logging
import json
import os
import shutil
from typing import Dict, Any, Optional, List
from dotenv import load_dotenv
import re
from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax
from rich import print as rprint

# --- Evolving Agents Imports ---
from evolving_agents.core.llm_service import LLMService
from evolving_agents.smart_library.smart_library import SmartLibrary
from evolving_agents.agent_bus.smart_agent_bus import SmartAgentBus
from evolving_agents.core.system_agent import SystemAgentFactory
from evolving_agents.core.dependency_container import DependencyContainer
from evolving_agents.firmware.firmware import Firmware
from evolving_agents.agents.architect_zero import ArchitectZeroAgentInitializer

# --- Configuration ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
load_dotenv()

# --- Rich Console for better display ---
console = Console()

# --- Constants ---
SMART_LIBRARY_PATH = "smart_library_demo.json"
AGENT_BUS_PATH = "smart_agent_bus_demo.json"
AGENT_BUS_LOG_PATH = "agent_bus_logs_demo.json"
VECTOR_DB_PATH = "./vector_db_demo"
CACHE_DIR = ".llm_cache_demo"
WORKFLOW_OUTPUT_PATH = "final_processing_output.json"
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
    console.print("[bold blue]OPERATION: Cleaning previous state...[/]")
    items_to_remove = [
        SMART_LIBRARY_PATH, AGENT_BUS_PATH, AGENT_BUS_LOG_PATH,
        VECTOR_DB_PATH, CACHE_DIR, WORKFLOW_OUTPUT_PATH, SAMPLE_INVOICE_PATH,
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
    console.print("[green]✓[/] Environment cleaned")

async def setup_framework_environment(container: DependencyContainer):
    """Initializes all core framework services and agents."""
    console.print("\n[bold blue]OPERATION: Initializing Framework Environment[/]")

    # 1. LLM Service
    console.print("  → Initializing LLM Service...")
    llm_service = LLMService(provider="openai", model="gpt-4o-mini", cache_dir=CACHE_DIR)
    container.register('llm_service', llm_service)

    # 2. Smart Library (with initial seeding)
    console.print("  → Initializing Smart Library...")
    smart_library = SmartLibrary(SMART_LIBRARY_PATH, llm_service=llm_service,
                                vector_db_path=VECTOR_DB_PATH, container=container)
    container.register('smart_library', smart_library)

    # 3. Firmware
    console.print("  → Initializing Firmware...")
    firmware = Firmware()
    container.register('firmware', firmware)

    # 4. Agent Bus
    console.print("  → Initializing Agent Bus...")
    agent_bus = SmartAgentBus(container=container, storage_path=AGENT_BUS_PATH,
                             log_path=AGENT_BUS_LOG_PATH, chroma_path=VECTOR_DB_PATH)
    container.register('agent_bus', agent_bus)

    # 5. System Agent
    console.print("  → Initializing System Agent...")
    system_agent = await SystemAgentFactory.create_agent(container=container)
    container.register('system_agent', system_agent)

    # 6. Architect Agent
    console.print("  → Initializing Architect Zero Agent...")
    architect_agent = await ArchitectZeroAgentInitializer.create_agent(container=container)
    container.register('architect_agent', architect_agent)

    # 7. Seed Library and Initialize Components
    console.print("  → Seeding initial library components...")
    await seed_initial_library(smart_library)
    await smart_library.initialize()

    # 8. Register components with Agent Bus
    console.print("  → Registering components with Agent Bus...")
    await agent_bus.initialize_from_library()

    console.print("[green]✓[/] Framework environment fully initialized")
    return container

async def seed_initial_library(smart_library: SmartLibrary):
    """Seed the library with basic components needed for context."""
    console.print("    - Adding BasicDocumentAnalyzer Tool...")
    # 1. Basic Document Analyzer Tool
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

    console.print("    - Adding BasicInvoiceProcessor Agent...")
    # 2. Basic Invoice Processor Agent
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

    console.print("    - Adding WeatherForecaster Tool (unrelated distraction)...")
    # 3. Weather Tool (unrelated, for distraction)
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

    console.print("    - Allowing time for vector database indexing...")
    await asyncio.sleep(3) # Allow some time for indexing

async def extract_json_with_llm(llm_service: LLMService, response_text: str) -> Optional[Dict[str, Any]]:
    """Use the LLM to extract JSON from agent response when pattern matching fails."""
    extraction_prompt = f"""
    Extract ONLY the JSON object from the following text.
    Return JUST the JSON object, with no additional explanation or text.
    If there's no valid JSON in the text, return a JSON object with
    a single "error" field explaining that no JSON was found.

    TEXT TO EXTRACT FROM:
    ```
    {response_text}
    ```
    """

    try:
        extracted_text = await llm_service.generate(extraction_prompt)
        # Try to parse the result
        try:
            # Basic cleaning for common LLM formatting issues
            cleaned_extracted_text = extracted_text.strip().strip('`').strip()
            if cleaned_extracted_text.startswith("json"):
                 cleaned_extracted_text = cleaned_extracted_text[len("json"):].strip()

            if cleaned_extracted_text.startswith("{") and cleaned_extracted_text.endswith("}"):
                return json.loads(cleaned_extracted_text)
            else:
                # If it doesn't look like JSON, log warning and return None
                logger.warning("LLM extraction did not produce a valid JSON object structure.")
                return None
        except json.JSONDecodeError:
            logger.warning(f"LLM extraction produced invalid JSON: {extracted_text}")
            return None
    except Exception as e:
        logger.error(f"Error using LLM to extract JSON: {e}")
        return None

def extract_json_from_response(response_text: str) -> Optional[Dict[str, Any]]:
    """Pattern matching to extract JSON from response text."""
    if not response_text: return None

    # 1. Try direct parsing of the whole text (cleaned)
    try:
        cleaned_text = response_text.strip()
        if cleaned_text.startswith("```json"): cleaned_text = cleaned_text[len("```json"):].strip()
        if cleaned_text.startswith("```"): cleaned_text = cleaned_text[len("```"):].strip()
        if cleaned_text.endswith("```"): cleaned_text = cleaned_text[:-len("```")].strip()
        if cleaned_text.startswith('{') and cleaned_text.endswith('}'):
            # Check balance of braces to increase confidence
            if cleaned_text.count('{') == cleaned_text.count('}'):
                return json.loads(cleaned_text)
    except json.JSONDecodeError: pass

    # 2. Look for ```json ... ``` blocks
    json_match = re.search(r"```json\s*([\s\S]*?)\s*```", response_text, re.MULTILINE)
    if json_match:
        try:
            potential_json = json_match.group(1).strip()
            if potential_json.startswith('{') and potential_json.endswith('}'):
                 if potential_json.count('{') == potential_json.count('}'):
                     return json.loads(potential_json)
        except json.JSONDecodeError: pass

    # 3. Look for ``` ... ``` blocks (might contain JSON)
    code_match = re.search(r"```\s*([\s\S]*?)\s*```", response_text, re.MULTILINE)
    if code_match:
        try:
            potential_json = code_match.group(1).strip()
            if potential_json.startswith('{') and potential_json.endswith('}'):
                 if potential_json.count('{') == potential_json.count('}'):
                     return json.loads(potential_json)
        except json.JSONDecodeError: pass

    # 4. Look for first '{' and last '}' (least reliable)
    start_index = response_text.find('{'); end_index = response_text.rfind('}')
    if start_index != -1 and end_index != -1 and end_index > start_index:
        potential_json = response_text[start_index : end_index + 1]
        try:
            # Check balance again
            if potential_json.count('{') == potential_json.count('}'):
                 # Basic check for common JSON structure elements
                 if ':' in potential_json and ('"' in potential_json or "'" in potential_json):
                     return json.loads(potential_json)
        except json.JSONDecodeError: pass

    logger.warning("Could not extract JSON using pattern matching.")
    return None

# --- Main Demo Function ---
async def run_demo():
    """Run the invoice processing demo to showcase SystemAgent capabilities."""
    console.print(Panel.fit(
        "[bold yellow]Evolving Agents Toolkit - Invoice Processing Demo[/]",
        border_style="yellow",
        padding=(1, 2)
    ))

    console.print("\n[bold]This demonstration shows how the SystemAgent can:[/]")
    console.print("  1. Accept a high-level business goal (invoice processing)")
    console.print("  2. Handle component search, creation, and orchestration internally")
    console.print("  3. Execute complex tasks without specifying implementation details")
    console.print("  4. Return structured, verified results\n")

    # --- Phase 0: Setup ---
    clean_previous_state()
    with open(SAMPLE_INVOICE_PATH, "w") as f: f.write(SAMPLE_INVOICE)
    console.print(f"\n[dim]Sample invoice saved to {SAMPLE_INVOICE_PATH}[/]")
    container = DependencyContainer()

    # --- Phase 1: Initialize Framework ---
    container = await setup_framework_environment(container)
    system_agent = container.get('system_agent')
    llm_service = container.get('llm_service')

    # --- Phase 2: Define & Execute High-Level Task for SystemAgent ---
    console.print("\n[bold blue]OPERATION: Defining and executing high-level task[/]")
    console.print("\n[bold]Task Description:[/] Process an invoice with these requirements:")
    console.print("  • Extract key fields (Invoice #, Date, Vendor, Bill To, Line Items, etc.)")
    console.print("  • Verify calculations (line items sum to subtotal, etc.)")
    console.print("  • Return structured JSON with verification results")

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

    console.print("\n[bold]Executing the task via SystemAgent...[/]")
    execution_result = {} # Initialize execution_result
    with console.status("[bold green]SystemAgent is working...", spinner="dots"):
        try:
            # Execute the SystemAgent to process the invoice
            sys_agent_response = await system_agent.run(high_level_task_prompt)
            final_output_text = sys_agent_response.result.text if hasattr(sys_agent_response.result, 'text') else str(sys_agent_response)

            # Try to extract JSON using pattern matching
            final_result_json = extract_json_from_response(final_output_text)

            # If pattern matching failed, try LLM extraction as a backup
            if not final_result_json:
                console.print("[yellow]⚠ Standard JSON extraction failed, trying LLM extraction...[/]")
                final_result_json = await extract_json_with_llm(llm_service, final_output_text)

            if final_result_json:
                final_result_data = final_result_json
                console.print("\r[green]✓[/] Successfully extracted JSON result from SystemAgent output") # Clear spinner line
            else:
                console.print("\r[red]✗[/] Could not extract JSON result, using raw text") # Clear spinner line
                final_result_data = final_output_text

            # Save the structured result
            execution_result = {
                "status": "completed" if isinstance(final_result_data, dict) else "completed_with_warnings",
                "final_result": final_result_data,
                "agent_full_output": final_output_text # Always store the raw text output
            }
            with open(WORKFLOW_OUTPUT_PATH, "w") as f:
                json.dump(execution_result, f, indent=2)

            console.print(f"[dim]Full output saved to {WORKFLOW_OUTPUT_PATH}[/]")

        except Exception as e:
            console.print(f"\r[bold red]Error executing task: {str(e)}[/]") # Clear spinner line
            execution_result = { "status": "error", "error_message": str(e), "agent_full_output": "" } # Ensure result is defined

    # --- Display the final result (Modified section) ---
    console.print("\n[bold green]OPERATION COMPLETE: Invoice Processing Result[/]")
    final_data_to_display = None
    if isinstance(execution_result.get("final_result"), dict):
        final_data_to_display = execution_result["final_result"]
        json_str = json.dumps(final_data_to_display, indent=2)
        syntax = Syntax(json_str, "json", theme="monokai", line_numbers=True)
        console.print(Panel(syntax, title="Extracted Invoice Data", border_style="green"))
    elif isinstance(execution_result.get("final_result"), str):
         # Attempt to parse if it's a string that might contain JSON
         try:
             parsed_json = extract_json_from_response(execution_result["final_result"])
             if parsed_json:
                 final_data_to_display = parsed_json
                 json_str = json.dumps(final_data_to_display, indent=2)
                 syntax = Syntax(json_str, "json", theme="monokai", line_numbers=True)
                 console.print(Panel(syntax, title="Extracted Invoice Data (Parsed from String)", border_style="yellow"))
             else:
                 # If it's just text, print it directly
                 console.print(execution_result["final_result"])
         except Exception:
             console.print(execution_result["final_result"]) # Print raw text on error
    else:
         # Print raw if not dict or string
         console.print(str(execution_result.get("final_result", "No result found.")))


    # Print a summary of key fields ONLY if we have a dictionary
    if final_data_to_display and isinstance(final_data_to_display, dict):
        console.print("\n[bold]Key Extracted Fields:[/]")
        # Use .get() with default 'N/A' for safety
        console.print(f"  • Invoice #: [bold]{final_data_to_display.get('invoice_number', final_data_to_display.get('InvoiceNumber', 'N/A'))}[/]") # Check common variations
        console.print(f"  • Vendor: [bold]{final_data_to_display.get('vendor', final_data_to_display.get('Vendor', 'N/A'))}[/]")
        console.print(f"  • Date: {final_data_to_display.get('date', final_data_to_display.get('Date', 'N/A'))}")
        console.print(f"  • Total: ${final_data_to_display.get('total_due', final_data_to_display.get('TotalDue', 'N/A'))}")

        # Print verification results
        verification = final_data_to_display.get('verification', {})
        status = verification.get('status', 'unknown')
        if status == 'ok':
            console.print(f"  • Verification: [bold green]PASSED[/]")
        else:
            console.print(f"  • Verification: [bold red]FAILED[/]") # Assume failed if not 'ok'
            discrepancies = verification.get('discrepancies', [])
            if discrepancies:
                for discrepancy in discrepancies:
                    console.print(f"    - {discrepancy}")
            elif status != 'unknown':
                 console.print(f"    - Reason: {verification.get('message', 'Details not provided')}") # Show message if status is known but not 'ok'
    else:
         console.print("\n[yellow]Could not extract structured data for key fields summary.[/]")

    # --- End of Modified Section ---

    console.print("\n[bold blue]DEMONSTRATION SUMMARY[/]")
    console.print("This demo showed the Evolving Agents Toolkit's ability to:")
    console.print("  1. [green]✓[/] Process a high-level business goal")
    console.print("  2. [green]✓[/] Dynamically manage components for the task")
    console.print("  3. [green]✓[/] Verify calculations in the extracted data")
    console.print("  4. [green]✓[/] Return structured JSON results")
    console.print("\nExternal files created:")
    console.print(f"  • [bold]{WORKFLOW_OUTPUT_PATH}[/]: Final structured results")
    console.print(f"  • [bold]{SMART_LIBRARY_PATH}[/]: Smart Library state")
    console.print(f"  • [bold]{AGENT_BUS_PATH}[/]: Agent Bus registry")
    console.print(f"  • [bold]{AGENT_BUS_LOG_PATH}[/]: Agent Bus execution logs")

# --- Run the Demo ---
if __name__ == "__main__":
    try:
        asyncio.run(run_demo())
    except Exception as main_error:
        console.print(f"[bold red]ERROR: Unhandled exception in main demo loop[/]")
        console.print(f"[red]{str(main_error)}[/]")
        import traceback
        traceback.print_exc() # Print full traceback for debugging