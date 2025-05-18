# /examples/agent_evolution/openai_agent_evolution_demo.py

import asyncio
import logging
import os
import sys
import json
import time
import re 
from typing import Dict, Any, List, Optional, Tuple
from dotenv import load_dotenv

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(script_dir)) 
sys.path.insert(0, project_root)

from evolving_agents.smart_library.smart_library import SmartLibrary
from evolving_agents.core.llm_service import LLMService
from evolving_agents.core.system_agent import SystemAgentFactory
from evolving_agents.agent_bus.smart_agent_bus import SmartAgentBus
from evolving_agents.providers.registry import ProviderRegistry
from evolving_agents.providers.beeai_provider import BeeAIProvider
from evolving_agents.providers.openai_agents_provider import OpenAIAgentsProvider
from evolving_agents.agents.agent_factory import AgentFactory
from evolving_agents.core.dependency_container import DependencyContainer
from evolving_agents.firmware.firmware import Firmware
from evolving_agents.core.mongodb_client import MongoDBClient
from evolving_agents import config as eat_config
from evolving_agents.utils.json_utils import safe_json_dumps


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

if not load_dotenv():
    logger.warning(".env file not found or python-dotenv not installed. Environment variables might not be loaded from .env.")

# --- Sample Data (remains the same) ---
SAMPLE_INVOICE = """
INVOICE #12345
Date: 2023-05-15
Vendor: TechSupplies Inc.
Address: 123 Business Ave, Commerce City, CA 90210
Items:
1. Laptop Computer - $1,200.00 (2 units)
2. Wireless Mouse - $25.00 (5 units)
3. External Hard Drive - $85.00 (3 units)
Subtotal: $1,680.00
Tax (8.5%): $142.80
Total Due: $1,822.80
Payment Terms: Net 30
Due Date: 2023-06-14
"""
SAMPLE_MEDICAL_RECORD = """
PATIENT MEDICAL RECORD
Patient ID: P789456
Name: John Smith
DOB: 1975-03-12
Visit Date: 2023-05-10
Chief Complaint: Patient presents with persistent cough for 2 weeks, mild fever, and fatigue.
Vitals:
- Temperature: 100.2°F
- Blood Pressure: 128/82
- Heart Rate: 88 bpm
- Respiratory Rate: 18/min
- Oxygen Saturation: 97%
Assessment: Acute bronchitis
Plan: Prescribed antibiotics (Azithromycin 500mg) for 5 days, recommended rest and increased fluid intake.
Follow-up in 1 week if symptoms persist.
"""
SAMPLE_CONTRACT = """
SERVICE AGREEMENT CONTRACT
Contract ID: CA-78901
Date: 2023-06-01
BETWEEN:
ABC Consulting Ltd. ("Provider")
123 Business Lane, Corporate City, BZ 54321
AND:
XYZ Corporation ("Client")
456 Commerce Ave, Enterprise Town, ET 12345
SERVICES:
Provider agrees to deliver the following services:
1. Strategic business consulting - 40 hours at $200/hour
2. Market analysis report - Fixed fee $5,000
3. Implementation support - 20 hours at $250/hour
TERM:
This agreement commences on July 1, 2023 and terminates on December 31, 2023.
PAYMENT TERMS:
- 30% deposit due upon signing
- 30% due upon delivery of market analysis report
- 40% due upon completion of implementation support
- All invoices due Net 15
TERMINATION:
Either party may terminate with 30 days written notice.
"""

# --- extract_component_id (can be simplified if we don't rely on it for create/evolve) ---
def extract_component_id_from_search(response_text: str) -> Optional[str]:
    """Extracts a component ID primarily from SearchComponentTool's JSON output."""
    if not response_text: return None
    try:
        json_match = re.search(r"(\{.*\})", response_text, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
            data = json.loads(json_str)
            if isinstance(data, dict) and "results" in data and isinstance(data["results"], list) and data["results"]:
                if isinstance(data["results"][0], dict) and "id" in data["results"][0]:
                    return data["results"][0].get("id")
    except Exception:
        pass # Fallback to general regex if specific search structure fails
    
    # Fallback regex for general ID patterns if needed
    patterns = [
        r'ID:\s*([a-fA-F0-9]{8}-(?:[a-fA-F0-9]{4}-){3}[a-fA-F0-9]{12}|[a-fA-F0-9]{24}|[a-zA-Z0-9_.\-]+)',
        r'([a-fA-F0-9]{8}-(?:[a-fA-F0-9]{4}-){3}[a-fA-F0-9]{12})',
    ]
    for pattern in patterns:
        match = re.search(pattern, response_text)
        if match and match.group(1):
            extracted_id = match.group(1).strip()
            if len(extracted_id) > 5 and not extracted_id.lower().startswith("http"):
                return extracted_id
    return None


async def use_llm_for_analysis(llm_service: LLMService, text: str, analysis_type: str) -> str:
    prompt = f"""
    Analyze the following text as a {analysis_type} document:
    {text}
    Please extract all key information and return a structured analysis.
    For invoices: Extract invoice number, date, vendor, items, costs, subtotal, tax, total.
    For medical records: Extract patient info, visit details, vitals, assessment, and plan.
    For contracts: Extract parties, services, terms, payment details, and termination conditions.
    Return your analysis in a structured, readable format.
    """
    result = await llm_service.generate(prompt)
    return result

async def text_analysis_demo(llm_service: LLMService, system_agent: Any, library: SmartLibrary): # Added library
    print("\n" + "-"*80)
    print("TEXT ANALYSIS DEMO: LLM SERVICE VS HARDCODED LOGIC")
    print("-"*80)
    
    print("\nAnalyzing invoice using LLM service...")
    start_time = time.time()
    llm_invoice_analysis = await use_llm_for_analysis(llm_service, SAMPLE_INVOICE, "invoice")
    llm_time = time.time() - start_time
    
    print("\nAnalyzing invoice using hardcoded regex logic...")
    start_time = time.time()
    hardcoded_analysis = {} # ... (regex logic remains the same)
    invoice_match = re.search(r'INVOICE #(\d+)', SAMPLE_INVOICE)
    if invoice_match: hardcoded_analysis["invoice_number"] = invoice_match.group(1)
    date_match = re.search(r'Date: ([\d-]+)', SAMPLE_INVOICE)
    if date_match: hardcoded_analysis["date"] = date_match.group(1)
    vendor_match = re.search(r'Vendor: ([^\n]+)', SAMPLE_INVOICE)
    if vendor_match: hardcoded_analysis["vendor"] = vendor_match.group(1)
    subtotal_match = re.search(r'Subtotal: \$([0-9,.]+)', SAMPLE_INVOICE)
    if subtotal_match: hardcoded_analysis["subtotal"] = subtotal_match.group(1)
    tax_match = re.search(r'Tax [^:]+: \$([0-9,.]+)', SAMPLE_INVOICE)
    if tax_match: hardcoded_analysis["tax"] = tax_match.group(1)
    total_match = re.search(r'Total Due: \$([0-9,.]+)', SAMPLE_INVOICE)
    if total_match: hardcoded_analysis["total"] = total_match.group(1)
    hardcoded_time = time.time() - start_time
    
    print("\nAnalyzing invoice using System Agent with SmartLibrary tools...")
    start_time = time.time()
    # Define the name of the tool we expect the SystemAgent to create if needed
    demo_invoice_tool_name = "DemoInvoiceProcessorTool_OpenAI_TextDemo" 
    system_prompt = f"""
    Your task is to analyze an invoice.
    1. Use the 'SearchComponentTool' with the query "invoice processing" to find components in the SmartLibrary.
    2. If suitable components are found, use the best one to process the invoice below.
    3. If no suitable component is found, use the 'CreateComponentTool' to create a new TOOL named "{demo_invoice_tool_name}" in the "finance" domain, framework "openai-agents". It should extract invoice number, date, vendor, items (name, price, quantity, total), subtotal, tax, and total_due. 
       IMPORTANT: If CreateComponentTool is used, your final response for THAT STEP should be ONLY its raw JSON output.
    4. After ensuring a tool exists (either found or created), use it to process the invoice.
    
    Invoice to process:
    ```
    {SAMPLE_INVOICE}
    ```
    Provide a final structured analysis of the invoice using the chosen/created tool.
    """
    system_response_message = await system_agent.run(system_prompt)
    system_response_text = system_response_message.result.text if hasattr(system_response_message, 'result') and hasattr(system_response_message.result, 'text') else str(system_response_message)
    system_time = time.time() - start_time
    
    # Check if the demo tool was created by the SystemAgent
    created_tool_record = await library.find_record_by_name(demo_invoice_tool_name, "TOOL")
    if created_tool_record:
        print(f"✓ SystemAgent created '{demo_invoice_tool_name}' with ID: {created_tool_record['id']}")
    
    print("\n" + "-"*30 + "\nANALYSIS COMPARISON\n" + "-"*30)
    print(f"\n1. LLM Analysis (took {llm_time:.2f}s):\n{llm_invoice_analysis[:500] + '...' if len(llm_invoice_analysis) > 500 else llm_invoice_analysis}")
    print(f"\n2. Hardcoded Regex Analysis (took {hardcoded_time:.2f}s):\n{json.dumps(hardcoded_analysis, indent=2)}")
    print(f"\n3. System Agent Analysis (took {system_time:.2f}s):\n{system_response_text[:500] + '...' if len(system_response_text) > 500 else system_response_text}")
    print("\n" + "-"*30 + "\nCOMPARISON INSIGHTS\n" + "-"*30) # ... (insights remain the same)
    print(f"""
    PERFORMANCE COMPARISON:
    - Hardcoded Regex: {hardcoded_time:.2f}s (fastest, but most limited)
    - LLM Service: {llm_time:.2f}s (slower, but flexible)
    - System Agent: {system_time:.2f}s (slowest, but most comprehensive)
    QUALITY COMPARISON: (Based on potential if tools work correctly)
    - Hardcoded Regex: Extracts only specifically programmed fields.
    - LLM Service: Can extract more information with better formatting.
    - System Agent: Can leverage SmartLibrary intelligence and provide comprehensive analysis by finding/creating/using specialized components.
    WHEN TO USE EACH:
    - Hardcoded Logic: For highly predictable, performance-critical tasks.
    - LLM Service: For flexible ad-hoc analysis without library overhead.
    - System Agent: For complex tasks requiring component reuse, evolution, and robust processing.
    """)

async def demonstrate_system_agent_tools(system_agent: Any, library: SmartLibrary, llm_service: LLMService):
    print("\n" + "-"*80 + "\nSYSTEM AGENT TOOLS DEMONSTRATION\n" + "-"*80)
    
    print("\n1. DEMONSTRATING SEARCH COMPONENT TOOL")
    search_prompt = """
    Use your SearchComponentTool to find components in the SmartLibrary that can process invoices.
    The query should be "invoice processing".
    Return all matching components with their similarity scores.
    If no components are found, indicate that clearly.
    IMPORTANT: Your final response should be ONLY the raw JSON output from the SearchComponentTool.
    """
    print("\nSearching for invoice processing components...")
    search_response_message = await system_agent.run(search_prompt)
    search_response_text = search_response_message.result.text if hasattr(search_response_message, 'result') and hasattr(search_response_message.result, 'text') else str(search_response_message)
    print("\nSearch results from SystemAgent:")
    print(search_response_text) 
    # The demo doesn't need to extract an ID from this search result, just observe.
    
    print("\n2. DEMONSTRATING CREATE COMPONENT TOOL")
    created_component_name = "AdvancedInvoiceExtractor_OpenAI_Demo"
    create_prompt = f"""
    Use your CreateComponentTool to create a new component called "{created_component_name}" 
    in the SmartLibrary.
    Component details:
    - record_type: TOOL
    - domain: finance
    - framework: openai-agents
    - description: OpenAI Tool to extract detailed invoice info, including line items, and validate calculations.
    - requirements: Must extract invoice number, date, vendor, line items (name, price, quantity, total for each), subtotal, tax, and total_due. Must validate subtotal + tax = total_due. Output should be clean JSON. Code should be suitable for OpenAI Agents SDK.
    IMPORTANT: Your final response for this creation step MUST be ONLY the raw JSON output from the CreateComponentTool.
    """
    print(f"\nCreating a new component: {created_component_name}...")
    create_response_message = await system_agent.run(create_prompt)
    create_response_text = create_response_message.result.text if hasattr(create_response_message, 'result') and hasattr(create_response_message.result, 'text') else str(create_response_message)
    print("\nComponent creation response from SystemAgent:")
    print(create_response_text)
    
    await asyncio.sleep(2) 
    created_record = await library.find_record_by_name(created_component_name, "TOOL")
    advanced_invoice_extractor_id = None
    if created_record:
        advanced_invoice_extractor_id = created_record["id"]
        print(f"✓ Verified: {created_component_name} created in library with ID: {advanced_invoice_extractor_id}")
    else:
        # Try extracting from text as a fallback, though less reliable
        extracted_id = extract_component_id_from_search(create_response_text) # Using the search-specific extractor for its JSON focus
        if extracted_id:
            advanced_invoice_extractor_id = extracted_id
            print(f"✓ Note: Extracted created ID from SystemAgent's text response: {advanced_invoice_extractor_id} (direct library verification preferred).")
        else:
            print(f"✗ Verification failed: Could not find {created_component_name} in library or extract ID from SystemAgent response.")


    component_name_to_evolve = "InvoiceProcessor_V1" # This is an AGENT
    initial_invoice_processor_record = await library.find_record_by_name(component_name_to_evolve, "AGENT")
    
    if initial_invoice_processor_record:
        initial_invoice_processor_id = initial_invoice_processor_record["id"]
        evolved_component_name = f"{component_name_to_evolve}_Evolved_OpenAI_Demo"
        print(f"\n3. DEMONSTRATING EVOLVE COMPONENT TOOL (evolving AGENT {component_name_to_evolve} ID: {initial_invoice_processor_id})")
        evolve_prompt = f"""
        Use your EvolveComponentTool to evolve the AGENT component with ID "{initial_invoice_processor_id}"
        into a new version named "{evolved_component_name}".
        Changes needed:
        - Better structured JSON output for its processing results.
        - Validation of calculations (subtotal + tax = total) within its logic.
        - Detection of due dates and payment terms.
        - Improved error handling.
        - Ensure it remains compatible with OpenAI Agents SDK.
        IMPORTANT: Your final response for this evolution step MUST be ONLY the raw JSON output from the EvolveComponentTool.
        """
        print(f"\nEvolving {component_name_to_evolve}...")
        evolve_response_message = await system_agent.run(evolve_prompt)
        evolve_response_text = evolve_response_message.result.text if hasattr(evolve_response_message, 'result') and hasattr(evolve_response_message.result, 'text') else str(evolve_response_message)
        print("\nComponent evolution response from SystemAgent:")
        print(evolve_response_text)

        await asyncio.sleep(2) 
        evolved_records = await library.components_collection.find({"parent_id": initial_invoice_processor_id, "name": evolved_component_name}).sort("created_at", -1).to_list(length=1)
        evolved_invoice_processor_id = None
        if evolved_records:
            evolved_invoice_processor_id = evolved_records[0]["id"]
            print(f"✓ Verified: {component_name_to_evolve} evolved in library. New component ID: {evolved_invoice_processor_id}, Name: {evolved_records[0]['name']}")
        else:
            extracted_evolved_id = extract_component_id_from_search(evolve_response_text) # Using the search-specific extractor for its JSON focus
            if extracted_evolved_id:
                 evolved_invoice_processor_id = extracted_evolved_id
                 print(f"✓ Note: Extracted evolved ID from SystemAgent's text response: {evolved_invoice_processor_id} (direct library verification preferred).")
            else:
                print(f"✗ Verification failed: Could not find evolved version of {component_name_to_evolve} in library or extract ID.")
    else:
        print(f"\n3. SKIPPING EVOLVE COMPONENT TOOL (component {component_name_to_evolve} not found)")
    
    print("\n4. USING SYSTEM AGENT WITH LIBRARY COMPONENTS")
    process_prompt = f"""
    Use the best available component in the SmartLibrary to process this invoice:
    ```
    {SAMPLE_INVOICE}
    ```
    First, use SearchComponentTool to find appropriate invoice processing components (prefer OpenAI agents if available).
    Then, use the most suitable one. Consider "{evolved_component_name if initial_invoice_processor_record else 'InvoiceProcessor_V1'}" or "{created_component_name if advanced_invoice_extractor_id else 'AdvancedInvoiceExtractor_OpenAI_Demo'}".
    Extract all information.
    If no suitable component exists, use CreateComponentTool to create one (prefer OpenAI framework), then use it.
    IMPORTANT: If you use CreateComponentTool or EvolveComponentTool, your final response is ONLY its raw JSON output. For execution, provide a summary.
    """
    print("\nProcessing invoice using SmartLibrary components...")
    process_response_message = await system_agent.run(process_prompt)
    process_response_text = process_response_message.result.text if hasattr(process_response_message, 'result') and hasattr(process_response_message.result, 'text') else str(process_response_message)
    print("\nProcessing result:")
    print(process_response_text)


async def setup_evolution_demo_library(library: SmartLibrary):
    print("\n[SMART LIBRARY DEMO SETUP]")
    demo_component_names = [
        "InvoiceProcessor_V1", "InvoiceProcessor_V1_Evolved_OpenAI_Demo", 
        "AdvancedInvoiceExtractor_OpenAI_Demo", "BasicInvoiceParser", 
        "SimpleContractAnalyzer", "MedicalRecordProcessor_OpenAI_Demo",
        "DemoInvoiceProcessorTool_OpenAI_TextDemo" # from text_analysis_demo
    ]
    delete_count = 0
    if library.components_collection is not None:
        for name in demo_component_names:
            result = await library.components_collection.delete_many({"name": name})
            if result.deleted_count > 0:
                delete_count += result.deleted_count
                print(f"    Cleaned up {result.deleted_count} records with name '{name}'.")
    print(f"  MongoDB: Cleaned up {delete_count} demo components.")
    print("Setting up initial OpenAI agent and tools for evolution demo...")
    
    await library.create_record(
        name="InvoiceProcessor_V1", record_type="AGENT", domain="finance",
        description="OpenAI agent for processing invoice documents",
        code_snippet="""from agents import Agent, Runner, ModelSettings\nagent = Agent(name="InvoiceProcessor", instructions='Extract key invoice details from the provided text. Focus on invoice number, date, vendor, items with quantities and prices, subtotal, tax, and total due. Format as structured JSON.', model="gpt-4o-mini")\nasync def p(t): return await Runner.run(agent, input=t)""",
        metadata={"framework": "openai-agents", "model": "gpt-4o-mini"}, tags=["openai", "invoice", "finance"]
    )
    print("✓ Created initial InvoiceProcessor_V1 agent")
    
    await library.create_record(
        name="BasicInvoiceParser", record_type="TOOL", domain="finance",
        description="Simple tool for parsing basic invoice information using regex patterns",
        code_snippet='import re\ndef p(t): return {"invoice_number": (m.group(1) if (m:=re.search(r"INVOICE #(\d+)",t)) else None)}',
        tags=["invoice", "parser", "finance", "regex"]
    )
    print("✓ Created BasicInvoiceParser tool")
    
    await library.create_record(
        name="SimpleContractAnalyzer", record_type="TOOL", domain="legal",
        description="Tool for extracting basic information from legal contracts",
        code_snippet='import re\ndef a(t): return {"parties": re.findall(r"([A-Z][A-Za-z\s]+(?:Ltd\.))[^\n]*",t)}',
        tags=["contract", "legal", "analysis"]
    )
    print("✓ Created SimpleContractAnalyzer tool")
    print(f"\nLibrary setup complete for OpenAI demo.")


class AgentExperienceTracker:
    def __init__(self, storage_path="openai_agent_experiences.json"):
        self.storage_path = storage_path
        self.experiences: Dict[str, Dict[str, Any]] = {}
        self._load_experiences()
    
    def _load_experiences(self):
        try:
            if os.path.exists(self.storage_path):
                with open(self.storage_path, 'r') as f:
                    self.experiences = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError): self.experiences = {}
        except Exception as e: logger.error(f"Error loading experiences: {e}"); self.experiences = {}

    def _save_experiences(self):
        try:
            with open(self.storage_path, 'w') as f: json.dump(self.experiences, f, indent=2)
        except Exception as e: logger.error(f"Error saving experiences: {e}")

    def record_invocation(self, agent_id: str, agent_name: str, domain: str, input_text: str, success: bool, response_time: float):
        if not agent_id: return
        if agent_id not in self.experiences:
            self.experiences[agent_id] = {"name": agent_name, "total_invocations": 0, "successful_invocations": 0, "domains": {}, "inputs": [], "evolutions": []}
        exp = self.experiences[agent_id]
        exp["total_invocations"] += 1
        if success: exp["successful_invocations"] += 1
        if domain not in exp["domains"]: exp["domains"][domain] = {"count": 0, "success": 0}
        exp["domains"][domain]["count"] += 1
        if success: exp["domains"][domain]["success"] += 1
        exp["inputs"].append({"text": input_text[:100]+"...", "success": success, "time": response_time, "timestamp": time.time()})
        exp["inputs"] = exp["inputs"][-10:]
        self._save_experiences()
    
    def record_evolution(self, agent_id: str, new_agent_id: str, evolution_type: str, changes: Dict[str, Any]):
        if not agent_id or not new_agent_id: return
        if agent_id not in self.experiences: self.record_invocation(agent_id, f"Original_of_{new_agent_id}", "unknown", "evolution_placeholder", True, 0)
        if "evolutions" not in self.experiences[agent_id]: self.experiences[agent_id]["evolutions"] = []
        self.experiences[agent_id]["evolutions"].append({"new_agent_id": new_agent_id, "evolution_type": evolution_type, "changes": changes, "timestamp": time.time()})
        self._save_experiences()
    
    def get_agent_experience(self, agent_id: str) -> Dict[str, Any]:
        if not agent_id: return {}
        return self.experiences.get(agent_id, {})

async def main():
    try:
        print("\n" + "="*80 + "\nOPENAI AGENT EVOLUTION DEMONSTRATION (MongoDB Backend)\n" + "="*80)
        
        container = DependencyContainer()
        
        try:
            mongo_uri = os.getenv("MONGODB_URI", eat_config.MONGODB_URI)
            mongo_db_name = os.getenv("MONGODB_DATABASE_NAME", eat_config.MONGODB_DATABASE_NAME)
            mongo_client = MongoDBClient(uri=mongo_uri, db_name=mongo_db_name) 
            await mongo_client.ping_server() 
            container.register('mongodb_client', mongo_client)
            print(f"✓ MongoDB Client initialized: DB='{mongo_client.db_name}'")
        except Exception as e:
            mongo_uri_log = os.getenv("MONGODB_URI", "Not Set") 
            mongo_db_name_log = os.getenv("MONGODB_DATABASE_NAME", "Not Set")
            logger.error(f"CRITICAL: MongoDBClient failed: {e}. URI: {mongo_uri_log}, DB: {mongo_db_name_log}")
            return

        llm_service = LLMService(provider=eat_config.LLM_PROVIDER, model=eat_config.LLM_MODEL,
                                 embedding_model=eat_config.LLM_EMBEDDING_MODEL, use_cache=eat_config.LLM_USE_CACHE, 
                                 container=container) 
        container.register('llm_service', llm_service)
        
        library = SmartLibrary(container=container)
        container.register('smart_library', library)
        await setup_evolution_demo_library(library) 

        firmware = Firmware(); container.register('firmware', firmware)
        agent_bus = SmartAgentBus(container=container); container.register('agent_bus', agent_bus)
        
        provider_registry = ProviderRegistry()
        provider_registry.register_provider(OpenAIAgentsProvider(llm_service)) 
        provider_registry.register_provider(BeeAIProvider(llm_service)) 
        container.register('provider_registry', provider_registry)
        
        agent_factory = AgentFactory(library, llm_service, provider_registry)
        container.register('agent_factory', agent_factory)
        
        print("\nInitializing System Agent...")
        system_agent = await SystemAgentFactory.create_agent(container=container)
        container.register('system_agent', system_agent)
        
        await library.initialize() 
        await agent_bus.initialize_from_library() 
        
        experience_tracker = AgentExperienceTracker()
        
        initial_agent_record = await library.find_record_by_name("InvoiceProcessor_V1", "AGENT")
        initial_agent_id = initial_agent_record["id"] if initial_agent_record else None
        if initial_agent_id: print(f"✓ Found InvoiceProcessor_V1 ID: {initial_agent_id}")
        else: logger.error("CRITICAL: InvoiceProcessor_V1 not found. Demo aborted."); return

        await text_analysis_demo(llm_service, system_agent, library) # Pass library
        await demonstrate_system_agent_tools(system_agent, library, llm_service)
        
        print("\n" + "-"*80 + "\nAGENT EVOLUTION VIA SYSTEM AGENT\n" + "="*80)
        evolved_agent_name = "InvoiceProcessor_V2_OpenAI_Demo"
        evolution_prompt = f"""
        Your task is to evolve the "InvoiceProcessor_V1" agent (ID: {initial_agent_id}).
        1. Use SearchComponentTool to find the agent with ID "{initial_agent_id}". (Summarize this step's findings).
        2. Then, use EvolveComponentTool to create "{evolved_agent_name}" with these improvements:
           - Better structured JSON output
           - Validation of calculations (subtotal + tax = total)
           - Detection of due dates and payment terms
           - Improved error handling
           - Ensure it remains compatible with OpenAI Agents SDK.
        IMPORTANT: Your final response for the evolution step (using EvolveComponentTool) MUST be ONLY the raw JSON output from that tool.
        """
        print("\nPrompting System Agent for evolution...")
        evolution_response_message = await system_agent.run(evolution_prompt)
        evolution_response_text = evolution_response_message.result.text if hasattr(evolution_response_message, 'result') and hasattr(evolution_response_message.result, 'text') else str(evolution_response_message)
        print("\nEvolution demonstration response from SystemAgent:")
        print(evolution_response_text)
        
        evolved_record = None
        await asyncio.sleep(2) 
        evolved_records_by_parent = await library.components_collection.find({"parent_id": initial_agent_id, "name": evolved_agent_name}).sort("created_at", -1).to_list(length=1)
        if evolved_records_by_parent:
            evolved_record = evolved_records_by_parent[0]
        if not evolved_record: 
            evolved_record = await library.find_record_by_name(evolved_agent_name, "AGENT")

        evolved_id = evolved_record["id"] if evolved_record else None
        
        if initial_agent_id and evolved_id and initial_agent_id != evolved_id:
            experience_tracker.record_evolution(
                initial_agent_id, evolved_id, "standard_evolution_openai", 
                {"changes": "Improved JSON, validation, due dates, error handling for OpenAI agent"}
            )
            print(f"✓ Verified and recorded evolution: {initial_agent_id} -> {evolved_id} ({evolved_record.get('name') if evolved_record else 'N/A'})")
        else: 
            extracted_evolved_id = extract_component_id_from_search(evolution_response_text)
            if extracted_evolved_id and extracted_evolved_id != initial_agent_id:
                 evolved_id = extracted_evolved_id
                 print(f"✓ Note: Extracted evolved ID from SystemAgent's text response: {evolved_id} (direct library verification preferred).")
                 experience_tracker.record_evolution( initial_agent_id, evolved_id, "standard_evolution_openai", {"changes": "Extracted from text"})
            else:
                print(f"✗ Could not verify or extract distinct evolved agent ID from library or SystemAgent response for tracking. Extracted from text: {extracted_evolved_id}")


        print("\n" + "-"*80 + "\nCROSS-DOMAIN ADAPTATION DEMONSTRATION\n" + "="*80)
        adapted_agent_name = "MedicalRecordProcessor_OpenAI_Demo"
        source_agent_id_for_adaptation = evolved_id if evolved_id and evolved_id != initial_agent_id else initial_agent_id

        adaptation_prompt = f"""
        Adapt an existing document processing agent for medical records.
        1. Use SearchComponentTool to find agent with ID "{source_agent_id_for_adaptation}". (Summarize findings).
        2. Use EvolveComponentTool with a 'domain_adaptation' strategy to create a new AGENT named "{adapted_agent_name}".
           Target domain: "medical".
           Framework: "openai-agents".
           Changes: Adapt to extract patient ID, name, DOB, visit date, chief complaint, vitals, assessment, and plan from medical records.
           It should process text like: "{SAMPLE_MEDICAL_RECORD[:150]}..."
        IMPORTANT: Your final response for the EvolveComponentTool step MUST be ONLY its raw JSON output.
        """
        print("\nPrompting System Agent for cross-domain adaptation...")
        adaptation_response_message = await system_agent.run(adaptation_prompt)
        adaptation_response_text = adaptation_response_message.result.text if hasattr(adaptation_response_message, 'result') and hasattr(adaptation_response_message.result, 'text') else str(adaptation_response_message)
        print("\nCross-domain adaptation response from SystemAgent:")
        print(adaptation_response_text)

        await asyncio.sleep(2)
        adapted_record = await library.find_record_by_name(adapted_agent_name, "AGENT")
        medical_processor_id = adapted_record["id"] if adapted_record else None
        
        if medical_processor_id:
            print(f"✓ Verified: {adapted_agent_name} adapted/created in library with ID: {medical_processor_id}")
            experience_tracker.record_evolution(
                source_agent_id_for_adaptation, medical_processor_id, "domain_adaptation_openai",
                {"target_domain": "medical", "original_framework": "openai-agents"}
            )
            # Test it 
            test_medical_prompt = f"Use the agent with ID '{medical_processor_id}' (which should be {adapted_agent_name}) to process: {SAMPLE_MEDICAL_RECORD}"
            test_med_res_msg = await system_agent.run(test_medical_prompt)
            print(f"\nTest of {adapted_agent_name}:\n{test_med_res_msg.result.text if hasattr(test_med_res_msg, 'result') else str(test_med_res_msg)}")
        else:
            extracted_medical_id = extract_component_id_from_search(adaptation_response_text)
            if extracted_medical_id:
                medical_processor_id = extracted_medical_id
                print(f"✓ Note: Extracted medical processor ID from SystemAgent's text response: {medical_processor_id} (direct library verification preferred).")
                experience_tracker.record_evolution(source_agent_id_for_adaptation, medical_processor_id, "domain_adaptation_openai", {"target_domain": "medical", "extracted_from_text": True})

            else:
                print(f"✗ Could not verify or extract {adapted_agent_name} ID.")
            
        print("\nDemonstration completed successfully!")
        
    except Exception as e:
        logger.error(f"Error in main demo: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # ... (dotenv loading remains the same) ...
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(script_dir))
    dotenv_path = os.path.join(project_root, '.env') 
    if not os.path.exists(dotenv_path):
        logger.warning(f".env file not found at {dotenv_path}. Trying CWD...")
        dotenv_path_cwd = os.path.join(os.getcwd(), '.env')
        if os.path.exists(dotenv_path_cwd): load_dotenv(dotenv_path_cwd); logger.info(f"Loaded .env from CWD: {dotenv_path_cwd}")
        else: logger.error(f"No .env at {dotenv_path} or {dotenv_path_cwd}.")
    else: load_dotenv(dotenv_path); logger.info(f"Loaded .env from: {dotenv_path}")

    if not os.getenv("MONGODB_URI") or not os.getenv("OPENAI_API_KEY"):
        logger.error("CRITICAL: MONGODB_URI and/or OPENAI_API_KEY missing.")
        logger.error(f"MONGODB_URI: {os.getenv('MONGODB_URI')}, OPENAI_API_KEY: {'SET' if os.getenv('OPENAI_API_KEY') else 'NOT SET'}")
    else:
        asyncio.run(main())