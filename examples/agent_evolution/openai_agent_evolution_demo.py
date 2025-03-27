import asyncio
import logging
import os
import sys
import json
import time
from typing import Dict, Any, List, Optional

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Sample invoice for testing
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

# Sample medical record for domain adaptation testing
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

# Sample contract for alternative domain adaptation
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

async def use_llm_for_analysis(llm_service: LLMService, text: str, analysis_type: str) -> str:
    """
    Use LLM service to analyze text instead of hardcoded logic
    
    Args:
        llm_service: The LLM service to use
        text: The text to analyze
        analysis_type: Type of analysis to perform
        
    Returns:
        Analysis result as string
    """
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

async def text_analysis_demo(llm_service: LLMService, system_agent):
    """
    Demonstrate text analysis using LLM service vs hardcoded logic
    
    Args:
        llm_service: The LLM service to use
        system_agent: The system agent
    """
    print("\n" + "-"*80)
    print("TEXT ANALYSIS DEMO: LLM SERVICE VS HARDCODED LOGIC")
    print("-"*80)
    
    # 1. LLM-based invoice analysis
    print("\nAnalyzing invoice using LLM service...")
    start_time = time.time()
    llm_invoice_analysis = await use_llm_for_analysis(llm_service, SAMPLE_INVOICE, "invoice")
    llm_time = time.time() - start_time
    
    # 2. Simple regex-based invoice analysis (hardcoded)
    print("\nAnalyzing invoice using hardcoded regex logic...")
    start_time = time.time()
    
    # Simple regex extraction
    import re
    hardcoded_analysis = {}
    
    # Extract invoice number
    invoice_match = re.search(r'INVOICE #(\d+)', SAMPLE_INVOICE)
    if invoice_match:
        hardcoded_analysis["invoice_number"] = invoice_match.group(1)
    
    # Extract date
    date_match = re.search(r'Date: ([\d-]+)', SAMPLE_INVOICE)
    if date_match:
        hardcoded_analysis["date"] = date_match.group(1)
    
    # Extract vendor
    vendor_match = re.search(r'Vendor: ([^\n]+)', SAMPLE_INVOICE)
    if vendor_match:
        hardcoded_analysis["vendor"] = vendor_match.group(1)
    
    # Extract totals
    subtotal_match = re.search(r'Subtotal: \$([0-9,.]+)', SAMPLE_INVOICE)
    if subtotal_match:
        hardcoded_analysis["subtotal"] = subtotal_match.group(1)
    
    tax_match = re.search(r'Tax [^:]+: \$([0-9,.]+)', SAMPLE_INVOICE)
    if tax_match:
        hardcoded_analysis["tax"] = tax_match.group(1)
    
    total_match = re.search(r'Total Due: \$([0-9,.]+)', SAMPLE_INVOICE)
    if total_match:
        hardcoded_analysis["total"] = total_match.group(1)
    
    hardcoded_time = time.time() - start_time
    
    # 3. System agent approach (using tools and SmartLibrary)
    print("\nAnalyzing invoice using System Agent with SmartLibrary tools...")
    start_time = time.time()
    
    system_prompt = """
    Analyze this invoice document using your SmartLibrary tools. 
    First, search for any existing invoice processing components in the library.
    If you don't find appropriate components, create a new one for invoice analysis.
    Then use the best available component to process this invoice:
    
    ```
    """ + SAMPLE_INVOICE + """
    ```
    
    Extract all key information and provide a structured analysis.
    """
    
    system_response = await system_agent.run(system_prompt)
    system_time = time.time() - start_time
    
    # Print comparison
    print("\n" + "-"*30)
    print("ANALYSIS COMPARISON")
    print("-"*30)
    print(f"\n1. LLM Analysis (took {llm_time:.2f}s):")
    print(llm_invoice_analysis[:500] + "..." if len(llm_invoice_analysis) > 500 else llm_invoice_analysis)
    
    print(f"\n2. Hardcoded Regex Analysis (took {hardcoded_time:.2f}s):")
    print(json.dumps(hardcoded_analysis, indent=2))
    
    print(f"\n3. System Agent Analysis (took {system_time:.2f}s):")
    print(system_response.result.text[:500] + "..." if len(system_response.result.text) > 500 else system_response.result.text)
    
    # Analyze and provide insights
    print("\n" + "-"*30)
    print("COMPARISON INSIGHTS")
    print("-"*30)
    
    insights = f"""
    PERFORMANCE COMPARISON:
    - Hardcoded Regex: {hardcoded_time:.2f}s (fastest, but most limited)
    - LLM Service: {llm_time:.2f}s (slower, but flexible)
    - System Agent: {system_time:.2f}s (slowest, but most comprehensive)
    
    QUALITY COMPARISON:
    - Hardcoded Regex: Extracts only specifically programmed fields
    - LLM Service: Extracts more information with better formatting
    - System Agent: Leverages SmartLibrary intelligence and provides the most comprehensive analysis
    
    WHEN TO USE EACH:
    - Hardcoded Logic: When speed is critical and the format is consistent
    - LLM Service: When flexibility is needed without library overhead
    - System Agent: When you need the full power of component reuse and evolution
    """
    
    print(insights)

async def demonstrate_system_agent_tools(system_agent, library: SmartLibrary, llm_service: LLMService):
    """
    Demonstrate system agent using its tools to interact with SmartLibrary
    
    Args:
        system_agent: The system agent
        library: The smart library
        llm_service: The LLM service
    """
    print("\n" + "-"*80)
    print("SYSTEM AGENT TOOLS DEMONSTRATION")
    print("-"*80)
    
    # 1. Demonstrate SearchComponentTool
    print("\n1. DEMONSTRATING SEARCH COMPONENT TOOL")
    search_prompt = """
    Use your SearchComponentTool to find components in the SmartLibrary that can process invoices.
    Return all matching components with their similarity scores.
    If no components are found, indicate that clearly.
    """
    
    print("\nSearching for invoice processing components...")
    search_response = await system_agent.run(search_prompt)
    print("\nSearch results:")
    print(search_response.result.text)
    
    # 2. Demonstrate CreateComponentTool
    print("\n2. DEMONSTRATING CREATE COMPONENT TOOL")
    create_prompt = """
    Use your CreateComponentTool to create a new component called "AdvancedInvoiceExtractor" 
    in the SmartLibrary.
    
    The component should:
    - Be of type TOOL
    - Belong to the "finance" domain
    - Extract detailed information from invoices including line items
    - Validate mathematical calculations (subtotal + tax = total)
    - Return the extracted data in clean JSON format
    
    Provide complete code for this component.
    """
    
    print("\nCreating a new invoice extractor component...")
    create_response = await system_agent.run(create_prompt)
    print("\nComponent creation result:")
    print(create_response.result.text)
    
    # 3. Demonstrate EvolveComponentTool
    # First find a component to evolve
    component_name = "InvoiceProcessor_V1"
    component = await library.find_record_by_name(component_name, "AGENT")
    
    if component:
        print(f"\n3. DEMONSTRATING EVOLVE COMPONENT TOOL (evolving {component_name})")
        evolve_prompt = f"""
        Use your EvolveComponentTool to evolve the existing component "{component_name}"
        into a more advanced version.
        
        The evolved component should:
        - Have improved JSON formatting
        - Add calculation verification (subtotal + tax = total)
        - Handle multiple invoice formats
        - Add error detection logic
        - Improve performance
        
        Name the evolved component "{component_name}_Evolved".
        """
        
        print(f"\nEvolving {component_name}...")
        evolve_response = await system_agent.run(evolve_prompt)
        print("\nComponent evolution result:")
        print(evolve_response.result.text)
    else:
        print(f"\n3. SKIPPING EVOLVE COMPONENT TOOL (component {component_name} not found)")
    
    # 4. Use system agent to process a document using library components
    print("\n4. USING SYSTEM AGENT WITH LIBRARY COMPONENTS")
    process_prompt = """
    Please process this invoice using the best available component in the SmartLibrary:
    
    ```
    """ + SAMPLE_INVOICE + """
    ```
    
    First search for appropriate invoice processing components,
    then use the most suitable one to extract all information from this invoice.
    
    If no suitable component exists, please create one, then use it.
    """
    
    print("\nProcessing invoice using SmartLibrary components...")
    process_response = await system_agent.run(process_prompt)
    print("\nProcessing result:")
    print(process_response.result.text)

async def setup_evolution_demo_library():
    """Create a library with an initial OpenAI agent for the evolution demo"""
    library_path = "openai_evolution_demo.json"
    
    # Delete existing file if it exists
    if os.path.exists(library_path):
        os.remove(library_path)
        print(f"Deleted existing library at {library_path}")
    
    # Initialize
    library = SmartLibrary(library_path)
    llm_service = LLMService(provider="openai", model="gpt-4o")
    
    print("Setting up initial OpenAI agent for evolution demo...")
    
    # Create a few sample components in the library
    
    # 1. Create basic invoice processor agent
    await library.create_record(
        name="InvoiceProcessor_V1",
        record_type="AGENT",
        domain="finance",
        description="OpenAI agent for processing invoice documents",
        code_snippet="""
from agents import Agent, Runner, ModelSettings

# Create an OpenAI agent for invoice processing
agent = Agent(
    name="InvoiceProcessor",
    instructions=\"\"\"
You are an invoice processing assistant that can extract information from invoice documents.

Extract the following fields:
- Invoice number
- Date
- Vendor name
- Items and prices
- Subtotal, tax, and total

Format your response in a clear, structured way.
\"\"\",
    model="gpt-4o",
    model_settings=ModelSettings(
        temperature=0.3
    )
)

# Helper function to process invoices
async def process_invoice(invoice_text):
    result = await Runner.run(agent, input=invoice_text)
    return result.final_output
""",
        metadata={
            "framework": "openai-agents",
            "model": "gpt-4o",
            "model_settings": {
                "temperature": 0.3
            }
        },
        tags=["openai", "invoice", "finance"]
    )
    print("✓ Created initial InvoiceProcessor_V1 agent")
    
    # 2. Create a basic invoice parser tool
    await library.create_record(
        name="BasicInvoiceParser",
        record_type="TOOL",
        domain="finance",
        description="Simple tool for parsing basic invoice information using regex patterns",
        code_snippet='''
import re
from typing import Dict, Any

def parse_invoice(invoice_text: str) -> Dict[str, Any]:
    """Parse invoice text using regex patterns to extract key information."""
    result = {}
    
    # Extract invoice number
    invoice_match = re.search(r'INVOICE #(\d+)', invoice_text, re.IGNORECASE)
    if invoice_match:
        result["invoice_number"] = invoice_match.group(1)
    
    # Extract date
    date_match = re.search(r'Date:?\s*([\d-/]+)', invoice_text, re.IGNORECASE)
    if date_match:
        result["date"] = date_match.group(1)
    
    # Extract vendor
    vendor_match = re.search(r'Vendor:?\s*([^\n]+)', invoice_text, re.IGNORECASE)
    if vendor_match:
        result["vendor"] = vendor_match.group(1).strip()
    
    # Extract totals
    subtotal_match = re.search(r'Subtotal:?\s*\$?([0-9,.]+)', invoice_text, re.IGNORECASE)
    if subtotal_match:
        result["subtotal"] = subtotal_match.group(1)
    
    tax_match = re.search(r'Tax[^:]*:?\s*\$?([0-9,.]+)', invoice_text, re.IGNORECASE)
    if tax_match:
        result["tax"] = tax_match.group(1)
    
    total_match = re.search(r'Total[^:]*:?\s*\$?([0-9,.]+)', invoice_text, re.IGNORECASE)
    if total_match:
        result["total"] = total_match.group(1)
    
    return result
''',
        version="1.0.0",
        tags=["invoice", "parser", "finance", "regex"]
    )
    print("✓ Created BasicInvoiceParser tool")
    
    # 3. Create a contract analyzer tool
    await library.create_record(
        name="SimpleContractAnalyzer",
        record_type="TOOL",
        domain="legal",
        description="Tool for extracting basic information from legal contracts",
        code_snippet='''
import re
from typing import Dict, Any

def analyze_contract(contract_text: str) -> Dict[str, Any]:
    """Extract key information from a legal contract."""
    result = {
        "parties": [],
        "dates": {},
        "payment_terms": [],
        "services": []
    }
    
    # Extract parties
    party_matches = re.findall(r'([A-Z][A-Za-z\s]+(?:Ltd\.|Inc\.|Corp\.|Corporation|Company)[^\n]*)', contract_text)
    for party in party_matches:
        result["parties"].append(party.strip())
    
    # Extract dates
    date_matches = re.findall(r'(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},\s+\d{4}|(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[.\s]+\d{1,2}[,.\s]+\d{4}|\d{1,2}[/-]\d{1,2}[/-]\d{2,4}', contract_text)
    if date_matches:
        result["dates"]["found_dates"] = date_matches
    
    # Look for specific date types
    effective_match = re.search(r'(?:effective|commencement|commence|commences|start|beginning)(?:\s+date)?(?:\s+on)?(?:\s+of)?(?:\s+is)?[:\s]+([^\.]+)', contract_text, re.IGNORECASE)
    if effective_match:
        result["dates"]["effective_date"] = effective_match.group(1).strip()
    
    termination_match = re.search(r'(?:terminat(?:es|ion)|expir(?:es|ation)|end(?:ing|s)?)(?:\s+date)?(?:\s+on)?(?:\s+of)?(?:\s+is)?[:\s]+([^\.]+)', contract_text, re.IGNORECASE)
    if termination_match:
        result["dates"]["termination_date"] = termination_match.group(1).strip()
    
    # Extract payment terms
    payment_match = re.search(r'Payment\s+Terms?[:\s]+([^\n.]+)', contract_text, re.IGNORECASE)
    if payment_match:
        result["payment_terms"].append(payment_match.group(1).strip())
    
    # Look for NET X payment terms
    net_terms = re.findall(r'Net[:\s]+(\d+)', contract_text, re.IGNORECASE)
    if net_terms:
        result["payment_terms"].extend([f"Net {term}" for term in net_terms])
    
    # Extract services or deliverables
    services_section = re.search(r'(?:SERVICES|DELIVERABLES)[:\s]+(.*?)(?=\n\s*\n[A-Z]+\s*:|$)', contract_text, re.DOTALL | re.IGNORECASE)
    if services_section:
        services_text = services_section.group(1).strip()
        services_list = re.findall(r'\d+\.\s*([^\n]+)', services_text)
        result["services"] = services_list if services_list else [services_text]
    
    return result
''',
        version="1.0.0",
        tags=["contract", "legal", "analysis"]
    )
    print("✓ Created SimpleContractAnalyzer tool")
    
    print(f"\nLibrary setup complete at: {library_path}")
    return library_path

# Simple agent experience tracker to record agent performance
class AgentExperienceTracker:
    def __init__(self, storage_path="agent_experiences.json"):
        self.storage_path = storage_path
        self.experiences = {}
        self._load_experiences()
    
    def _load_experiences(self):
        try:
            if os.path.exists(self.storage_path):
                with open(self.storage_path, 'r') as f:
                    self.experiences = json.load(f)
        except:
            self.experiences = {}
    
    def _save_experiences(self):
        with open(self.storage_path, 'w') as f:
            json.dump(self.experiences, f, indent=2)
    
    def record_invocation(self, agent_id, agent_name, domain, input_text, success, response_time):
        if agent_id not in self.experiences:
            self.experiences[agent_id] = {
                "name": agent_name,
                "total_invocations": 0,
                "successful_invocations": 0,
                "domains": {},
                "inputs": []
            }
        
        exp = self.experiences[agent_id]
        exp["total_invocations"] += 1
        if success:
            exp["successful_invocations"] += 1
            
        if domain not in exp["domains"]:
            exp["domains"][domain] = {"count": 0, "success": 0}
        
        exp["domains"][domain]["count"] += 1
        if success:
            exp["domains"][domain]["success"] += 1
            
        # Store recent inputs (keep last 10)
        exp["inputs"].append({
            "text": input_text[:100] + "...",
            "success": success,
            "time": response_time,
            "timestamp": time.time()
        })
        if len(exp["inputs"]) > 10:
            exp["inputs"] = exp["inputs"][-10:]
            
        self._save_experiences()
    
    def record_evolution(self, agent_id, new_agent_id, evolution_type, changes):
        if agent_id not in self.experiences:
            return
            
        if "evolutions" not in self.experiences[agent_id]:
            self.experiences[agent_id]["evolutions"] = []
            
        self.experiences[agent_id]["evolutions"].append({
            "new_agent_id": new_agent_id,
            "evolution_type": evolution_type,
            "changes": changes,
            "timestamp": time.time()
        })
        
        self._save_experiences()
    
    def get_agent_experience(self, agent_id):
        return self.experiences.get(agent_id, {})

async def main():
    try:
        print("\n" + "="*80)
        print("ENHANCED SYSTEM AGENT AND SMART LIBRARY DEMONSTRATION")
        print("="*80)
        
        # Initialize components with dependency container
        container = DependencyContainer()
        
        # Set up library
        library_path = await setup_evolution_demo_library()
        
        # Initialize services
        llm_service = LLMService(provider="openai", model="gpt-4o")
        container.register('llm_service', llm_service)
        
        library = SmartLibrary(library_path, container=container)
        container.register('smart_library', library)
        
        # Create firmware component
        firmware = Firmware()
        container.register('firmware', firmware)
        
        # Create agent bus
        agent_bus = SmartAgentBus(
            storage_path="smart_agent_bus.json", 
            log_path="agent_bus_logs.json",
            container=container
        )
        container.register('agent_bus', agent_bus)
        
        # Set up provider registry
        provider_registry = ProviderRegistry()
        provider_registry.register_provider(BeeAIProvider(llm_service))
        provider_registry.register_provider(OpenAIAgentsProvider(llm_service))
        container.register('provider_registry', provider_registry)
        
        # Create agent factory
        agent_factory = AgentFactory(library, llm_service, provider_registry)
        container.register('agent_factory', agent_factory)
        
        # Initialize System Agent
        print("\nInitializing System Agent...")
        system_agent = await SystemAgentFactory.create_agent(container=container)
        container.register('system_agent', system_agent)
        
        # Initialize components
        await library.initialize()
        await agent_bus.initialize_from_library()
        
        # Create agent experience tracker
        experience_tracker = AgentExperienceTracker()
        
        # Get the initial agent record
        initial_agent_record = await library.find_record_by_name("InvoiceProcessor_V1", "AGENT")
        if initial_agent_record:
            initial_agent_id = initial_agent_record["id"]
            print(f"✓ Found InvoiceProcessor_V1 agent with ID: {initial_agent_id}")
        
        # DEMONSTRATION 1: Text Analysis Comparison
        await text_analysis_demo(llm_service, system_agent)
        
        # DEMONSTRATION 2: System Agent Tools
        await demonstrate_system_agent_tools(system_agent, library, llm_service)
        
        # DEMONSTRATION 3: Evolution via System Agent
        print("\n" + "-"*80)
        print("AGENT EVOLUTION VIA SYSTEM AGENT")
        print("-"*80)
        
        evolution_prompt = """
        Please demonstrate the process of evolving the "InvoiceProcessor_V1" agent using the following steps:
        
        1. First, search for the agent in the SmartLibrary using your SearchComponentTool
        2. Analyze the agent's current capabilities and architecture
        3. Create an evolved "InvoiceProcessor_V2" agent using your CreateComponentTool with these improvements:
           - Better structured JSON output
           - Validation of calculations (subtotal + tax = total)
           - Detection of due dates and payment terms
           - Improved error handling
        4. Compare the original and evolved agents, highlighting the differences
        
        Please clearly explain what you're doing at each step, showing how the tools work.
        """
        
        print("\nPrompting System Agent to demonstrate evolution process...")
        evolution_response = await system_agent.run(evolution_prompt)
        
        print("\nEvolution demonstration:")
        print(evolution_response.result.text)
        
        # DEMONSTRATION 4: Cross-Domain Adaptation
        print("\n" + "-"*80)
        print("CROSS-DOMAIN ADAPTATION DEMONSTRATION")
        print("-"*80)
        
        adaptation_prompt = f"""
        Please demonstrate how to adapt document processing across different domains using the SmartLibrary.
        
        We have:
        - Invoice processors in the finance domain
        - A need to process documents in the medical and legal domains
        
        Using your tools, please:
        
        1. Analyze the components we already have in the SmartLibrary
        2. Show how to adapt an invoice processor to handle medical records
        3. Create a new MedicalRecordProcessor based on invoice processing techniques
        4. Test the new processor on this sample: {SAMPLE_MEDICAL_RECORD}
        5. Explain the adaptation strategies you used
        
        Please clearly explain each step of your process.
        """
        
        print("\nPrompting System Agent to demonstrate cross-domain adaptation...")
        adaptation_response = await system_agent.run(adaptation_prompt)
        
        print("\nCross-domain adaptation demonstration:")
        print(adaptation_response.result.text)
        
        print("\nDemonstration completed successfully!")
        
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())