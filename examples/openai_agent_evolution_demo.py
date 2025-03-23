# examples/openai_agent_evolution_demo.py

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

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

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
    
    # Create initial invoice processor agent
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
        print("OPENAI AGENT EVOLUTION DEMONSTRATION")
        print("="*80)
        
        # Initialize components
        library_path = await setup_evolution_demo_library()
        library = SmartLibrary(library_path)
        llm_service = LLMService(provider="openai", model="gpt-4o")
        agent_bus = SmartAgentBus(
            smart_library=library,
            system_agent=None,  # We'll set this after system_agent is created
            storage_path="smart_agent_bus.json", 
            log_path="agent_bus_logs.json"
        )
        
        # Set up provider registry
        provider_registry = ProviderRegistry()
        provider_registry.register_provider(BeeAIProvider(llm_service))
        provider_registry.register_provider(OpenAIAgentsProvider(llm_service))
        
        # Create agent factory
        agent_factory = AgentFactory(library, llm_service, provider_registry)
        
        # Initialize System Agent
        system_agent = await SystemAgentFactory.create_agent(
            llm_service=llm_service,
            smart_library=library,
            agent_bus=agent_bus
        )

        # Update the agent_bus with the system_agent
        agent_bus.system_agent = system_agent

        # Initialize from library
        await agent_bus.initialize_from_library()
        
        # Create agent experience tracker
        experience_tracker = AgentExperienceTracker()
        
        # Get the initial agent record
        initial_agent_record = await library.find_record_by_name("InvoiceProcessor_V1", "AGENT")
        initial_agent_id = initial_agent_record["id"]
        
        # PHASE 1: Register and test the initial agent
        print("\n" + "-"*80)
        print("PHASE 1: REGISTER AND TEST INITIAL AGENT")
        print("-"*80)
        
        register_prompt = """
        Check if the agent 'InvoiceProcessor_V1' is registered with the Agent Bus.
        If not, please register it so it's available for use in our system.
        """
        
        print("\nRegistering the agent with the Agent Bus...")
        register_response = await system_agent.run(register_prompt)
        print(f"Registration result: {register_response.result.text}")
        
        # Test inputs
        test_inputs = [
            f"Process this invoice and extract all important information: {SAMPLE_INVOICE}",
            "Extract the vendor, date, and total from this invoice: Invoice #9876 from Acme Corp, dated 2023-04-22, for $543.21",
            "Is this a valid invoice? It has no total amount: Invoice #5555 from XYZ Services, dated 2023-06-10"
        ]
        
        print("\nTesting the agent with sample inputs...")
        
        for i, input_text in enumerate(test_inputs):
            test_prompt = f"""
            Use the InvoiceProcessor_V1 agent to process this input:
            
            {input_text}
            
            Extract all relevant information and return the results.
            """
            
            print(f"\nTest {i+1}: Processing invoice...")
            start_time = time.time()
            test_response = await system_agent.run(test_prompt)
            response_time = time.time() - start_time
            
            # Record the experience
            experience_tracker.record_invocation(
                initial_agent_id,
                initial_agent_record["name"],
                "finance",
                input_text,
                True,  # Assuming success
                response_time
            )
            
            print(f"Response (in {response_time:.2f}s):")
            print(test_response.result.text[:300] + "..." if len(test_response.result.text) > 300 else test_response.result.text)
        
        # PHASE 2: Standard Evolution - Improving capabilities
        print("\n" + "-"*80)
        print("PHASE 2: STANDARD EVOLUTION - IMPROVING CAPABILITIES")
        print("-"*80)
        
        # Get the agent's experience
        agent_experience = experience_tracker.get_agent_experience(initial_agent_id)
        avg_response_time = sum([input_data["time"] for input_data in agent_experience.get("inputs", [])]) / max(1, len(agent_experience.get("inputs", [])))
        
        standard_evolution_prompt = f"""
        I need to evolve our InvoiceProcessor_V1 agent with the standard evolution strategy. 
        
        Current agent information:
        - Name: InvoiceProcessor_V1
        - Domain: finance
        - Framework: openai-agents
        - Experience: {agent_experience.get("total_invocations", 0)} invocations
        - Average response time: {avg_response_time:.2f} seconds
        
        Please create an evolved version with these improvements:
        1. Better structured output in JSON format
        2. Validation of calculations (subtotal + tax = total)
        3. Ability to handle multiple invoice formats
        4. Detection of due dates and payment terms
        
        The evolved agent should be named 'InvoiceProcessor_V2' and keep the same underlying framework.
        """
        
        print("\nPrompting SystemAgent to evolve the agent with standard strategy...")
        evolution_response = await system_agent.run(standard_evolution_prompt)
        
        print("\nEvolution result:")
        print(evolution_response.result.text)
        
        # Find the evolved agent
        evolved_agent_record = await library.find_record_by_name("InvoiceProcessor_V2", "AGENT")
        if evolved_agent_record:
            evolved_agent_id = evolved_agent_record["id"]
            print(f"✓ Found evolved agent: {evolved_agent_record['name']}")
            
            # Record the evolution
            experience_tracker.record_evolution(
                initial_agent_id, 
                evolved_agent_id, 
                "standard", 
                "Improved JSON output, calculation verification, multiple formats, recommendations"
            )
            
            # Test the evolved agent
            test_prompt = f"""
            Use the evolved InvoiceProcessor_V2 agent to process this invoice:
            
            {SAMPLE_INVOICE}
            
            I expect to see the improvements we made including structured JSON output, 
            validation of calculations, and any relevant recommendations.
            """
            
            print("\nTesting the evolved agent...")
            start_time = time.time()
            evolved_test_response = await system_agent.run(test_prompt)
            response_time = time.time() - start_time
            
            # Record the experience
            experience_tracker.record_invocation(
                evolved_agent_id,
                evolved_agent_record["name"],
                "finance",
                SAMPLE_INVOICE,
                True,
                response_time
            )
            
            print(f"\nEvolved agent response (in {response_time:.2f}s):")
            print(evolved_test_response.result.text[:300] + "..." if len(evolved_test_response.result.text) > 300 else evolved_test_response.result.text)
            
            # PHASE 3: Domain Adaptation - Medical Records
            print("\n" + "-"*80)
            print("PHASE 3: DOMAIN ADAPTATION - MEDICAL RECORDS")
            print("-"*80)
            
            domain_adaptation_prompt = f"""
            I need to adapt our evolved InvoiceProcessor_V2 agent to handle medical records using domain adaptation.
            
            Current agent:
            - Name: InvoiceProcessor_V2
            - Domain: finance
            - Purpose: processing invoices
            
            Target domain: healthcare
            
            Please create a domain-adapted agent that can:
            1. Extract patient information (name, ID, DOB)
            2. Extract visit details (date, reason for visit)
            3. Extract vital signs
            4. Extract assessment and plan
            
            Create a new agent called 'MedicalRecordProcessor_V1' with the appropriate domain.
            
            Sample medical record:
            {SAMPLE_MEDICAL_RECORD}
            """
            
            print("\nPrompting SystemAgent to adapt the agent to the medical domain...")
            adaptation_response = await system_agent.run(domain_adaptation_prompt)
            
            print("\nDomain adaptation result:")
            print(adaptation_response.result.text)
            
            # Find the medical agent
            medical_agent_record = await library.find_record_by_name("MedicalRecordProcessor_V1", "AGENT")
            if medical_agent_record:
                medical_agent_id = medical_agent_record["id"]
                print(f"✓ Found medical domain agent: {medical_agent_record['name']}")
                
                # Record the domain adaptation
                experience_tracker.record_evolution(
                    evolved_agent_id,
                    medical_agent_id,
                    "domain_adaptation",
                    "Adapted from invoice processing to medical record processing"
                )
                
                # Test the medical agent
                medical_test_prompt = f"""
                Use the MedicalRecordProcessor_V1 agent to process this medical record:
                
                {SAMPLE_MEDICAL_RECORD}
                
                Extract all relevant patient information, vital signs, assessment, and plan.
                """
                
                print("\nTesting the medical domain agent...")
                start_time = time.time()
                medical_test_response = await system_agent.run(medical_test_prompt)
                response_time = time.time() - start_time
                
                # Record the experience
                experience_tracker.record_invocation(
                    medical_agent_id,
                    medical_agent_record["name"],
                    "healthcare",
                    SAMPLE_MEDICAL_RECORD,
                    True,
                    response_time
                )
                
                print(f"\nMedical agent response (in {response_time:.2f}s):")
                print(medical_test_response.result.text[:300] + "..." if len(medical_test_response.result.text) > 300 else medical_test_response.result.text)
                
                # PHASE 4: Compare Agents - A/B Testing
                print("\n" + "-"*80)
                print("PHASE 4: COMPARE AGENTS - A/B TESTING")
                print("-"*80)
                
                comparison_prompt = f"""
                I need to compare the performance of our original InvoiceProcessor_V1 agent 
                with the evolved InvoiceProcessor_V2 agent.
                
                Please run both agents on the following invoice:
                {SAMPLE_INVOICE}
                
                Evaluate both outputs on these criteria:
                1. Accuracy of extraction
                2. Completeness of output
                3. Structured format
                4. Validation checks
                5. Response quality
                
                Provide a detailed comparison with scores for each criterion (1-10), 
                declare a winner, and explain why one is better than the other.
                """
                
                print("\nPrompting SystemAgent to compare original and evolved agents...")
                comparison_response = await system_agent.run(comparison_prompt)
                
                print("\nComparison result:")
                print(comparison_response.result.text)
                
                # PHASE 5: Aggressive Evolution
                print("\n" + "-"*80)
                print("PHASE 5: AGGRESSIVE EVOLUTION - CONTRACT ANALYZER")
                print("-"*80)
                
                aggressive_evolution_prompt = f"""
                I need to perform an aggressive evolution of our InvoiceProcessor_V2 agent to create 
                a completely new type of document processor - a ContractAnalyzer.
                
                This is a more radical change than standard evolution or domain adaptation.
                The new agent should:
                
                1. Parse and extract key details from legal contracts
                2. Identify parties, effective dates, and termination conditions
                3. Extract services, fees, and payment terms
                4. Highlight key obligations and potential legal risks
                5. Provide a legal compliance check
                
                Create a new agent called 'ContractAnalyzer_V1' with domain 'legal'
                and framework 'openai-agents'.
                
                Sample contract:
                {SAMPLE_CONTRACT}
                """
                
                print("\nPrompting SystemAgent to aggressively evolve the agent into a contract analyzer...")
                aggressive_response = await system_agent.run(aggressive_evolution_prompt)
                
                print("\nAggressive evolution result:")
                print(aggressive_response.result.text)
                
                # Find the contract agent
                contract_agent_record = await library.find_record_by_name("ContractAnalyzer_V1", "AGENT")
                if contract_agent_record:
                    contract_agent_id = contract_agent_record["id"]
                    print(f"✓ Found contract analyzer agent: {contract_agent_record['name']}")
                    
                    # Record the aggressive evolution
                    experience_tracker.record_evolution(
                        evolved_agent_id,
                        contract_agent_id,
                        "aggressive",
                        "Radically transformed from invoice processor to contract analyzer"
                    )
                    
                    # Test the contract agent
                    contract_test_prompt = f"""
                    Use the ContractAnalyzer_V1 agent to analyze this contract:
                    
                    {SAMPLE_CONTRACT}
                    
                    Extract all parties, terms, services, payment details, and highlight
                    any potential legal concerns.
                    """
                    
                    print("\nTesting the contract analyzer agent...")
                    start_time = time.time()
                    contract_test_response = await system_agent.run(contract_test_prompt)
                    response_time = time.time() - start_time
                    
                    # Record the experience
                    experience_tracker.record_invocation(
                        contract_agent_id,
                        contract_agent_record["name"],
                        "legal",
                        SAMPLE_CONTRACT,
                        True,
                        response_time
                    )
                    
                    print(f"\nContract analyzer response (in {response_time:.2f}s):")
                    print(contract_test_response.result.text[:300] + "..." if len(contract_test_response.result.text) > 300 else contract_test_response.result.text)
                
                # PHASE 6: Final Evolution Assessment
                print("\n" + "-"*80)
                print("PHASE 6: FINAL EVOLUTION ASSESSMENT")
                print("-"*80)
                
                assessment_prompt = """
                Please provide a comprehensive assessment of our agent evolution process.
                
                Our agents have gone through several types of evolution:
                1. Standard evolution: InvoiceProcessor_V1 → InvoiceProcessor_V2
                2. Domain adaptation: InvoiceProcessor_V2 → MedicalRecordProcessor_V1
                3. Aggressive evolution: InvoiceProcessor_V2 → ContractAnalyzer_V1
                
                For each evolution type:
                - Evaluate the effectiveness of the evolution approach
                - Identify strengths and weaknesses
                - Provide suggestions for improving the evolution process
                
                Also provide general remarks on how the OpenAI Agents SDK integration
                has enhanced our Evolving Agents Framework.
                """
                
                print("\nPrompting SystemAgent for a final evolution assessment...")
                assessment_response = await system_agent.run(assessment_prompt)
                
                print("\nFinal evolution assessment:")
                print(assessment_response.result.text)
            else:
                print("✗ Medical domain agent not found")
        else:
            print("✗ Evolved agent not found")
        
        print("\nEvolution demonstration completed successfully!")
        
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())