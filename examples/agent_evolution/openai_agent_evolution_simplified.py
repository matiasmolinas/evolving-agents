# examples/openai_agent_evolution_simplified.py

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

# Medical record for domain adaptation testing
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

# Simple agent experience logger (scaled-down version of the full implementation)
class SimpleAgentLogger:
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

async def main():
    try:
        print("\n" + "="*80)
        print("OPENAI AGENT EVOLUTION DEMONSTRATION (SIMPLIFIED)")
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
            agent_bus=agent_bus,
            memory_type="token"
        )
        
        # Create a simple agent logger
        agent_logger = SimpleAgentLogger()
        
        # Get the initial agent record
        initial_agent_record = await library.find_record_by_name("InvoiceProcessor_V1", "AGENT")
        initial_agent_id = initial_agent_record["id"]
        
        # *** Create the agent but don't try to register directly with the Agent Bus ***
        print("\nCreating the agent instance...")
        initial_agent = await agent_factory.create_agent(initial_agent_record)
        print("✓ Agent created successfully")
        
        # PHASE 1: Process invoices with the initial agent and record experiences
        print("\n" + "-"*80)
        print("PHASE 1: PROCESS INVOICES WITH INITIAL AGENT")
        print("-"*80)
        
        # Sample inputs to process
        test_inputs = [
            f"Extract information from this invoice: {SAMPLE_INVOICE[:300]}...",
            "Extract the vendor, date, and total from this invoice: Invoice #9876 from Acme Corp, dated 2023-04-22, for $543.21"
        ]
        
        print("\nProcessing test inputs with initial agent using the SystemAgent...")
        
        # Process each input and log the experience using the SystemAgent instead
        for i, input_text in enumerate(test_inputs):
            print(f"\nInput {i+1}: {input_text[:50]}...")
            
            # Use the SystemAgent to process the request
            start_time = time.time()
            prompt = f"""
            I need to process this input with our InvoiceProcessor_V1 agent.
            
            Input: {input_text}
            
            Please extract all relevant information and process this as our invoice agent would.
            """
            response = await system_agent.run(prompt)
            response_time = time.time() - start_time
            
            # Log the experience
            success = True
            agent_logger.record_invocation(
                initial_agent_id,
                initial_agent_record["name"],
                "finance",
                input_text,
                success,
                response_time
            )
            
            print(f"  Response time: {response_time:.2f}s")
            print(f"  Result: {response.result.text[:150]}...")
        
        # PHASE 2: Use System Agent to evolve the agent
        print("\n" + "-"*80)
        print("PHASE 2: SYSTEM AGENT EVOLVES THE AGENT")
        print("-"*80)
        
        evolution_prompt = f"""
        I need you to evolve our OpenAI invoice processing agent to handle more complex cases.

        Current agent: InvoiceProcessor_V1
        Domain: finance
        
        Please create a new improved agent with the following features:
        1. Improved ability to extract structured data from invoices in JSON format
        2. Calculation verification to check if subtotal + tax = total
        3. Support for multiple invoice formats
        4. Recommendations based on the invoice content (e.g., payment due soon)
        
        Create a new agent record in the library with improved instructions and capabilities.
        The new agent should have a version number V2 and include the openai-agents framework.
        """
        
        print("\nPrompting SystemAgent to evolve the invoice agent...")
        evolution_response = await system_agent.run(evolution_prompt)
        
        print("\nSystemAgent's evolution response:")
        print(evolution_response.result.text)
        
        print("\nChecking if a new agent was created...")
        # Look for any new agents that might have been created
        all_agents = [r for r in library.records if r["record_type"] == "AGENT" and r["id"] != initial_agent_id]
        
        if all_agents:
            evolved_agent = all_agents[-1]  # Get the most recently created agent
            print(f"✓ Found evolved agent: {evolved_agent['name']}")
            
            # Log the evolution
            agent_logger.record_evolution(
                initial_agent_id, 
                evolved_agent["id"], 
                "standard", 
                "Improved JSON output, calculation verification, multiple formats, recommendations"
            )
            
            # PHASE 3: Test with medical documents (domain adaptation)
            print("\n" + "-"*80)
            print("PHASE 3: DOMAIN ADAPTATION TEST")
            print("-"*80)
            
            adaptation_prompt = f"""
            I want to adapt our invoice processing agent to handle medical records instead.
            
            Here's a sample medical record:
            {SAMPLE_MEDICAL_RECORD}
            
            Create a new agent specialized for medical records that can extract:
            - Patient information (name, ID, DOB)
            - Visit details
            - Vital signs
            - Assessment and plan
            
            This requires adapting from finance domain to healthcare domain.
            The new agent should be called MedicalRecordProcessor_V1 with framework openai-agents.
            """
            
            print("\nPrompting SystemAgent to adapt the agent to the medical domain...")
            adaptation_response = await system_agent.run(adaptation_prompt)
            
            print("\nSystemAgent's domain adaptation response:")
            print(adaptation_response.result.text)
            
            # Check for the new medical agent
            medical_agents = [r for r in library.records if r["record_type"] == "AGENT" and "medical" in r["name"].lower()]
            if medical_agents:
                medical_agent = medical_agents[-1]
                print(f"✓ Found medical agent: {medical_agent['name']}")
                
                # Log the domain adaptation
                agent_logger.record_evolution(
                    evolved_agent["id"],
                    medical_agent["id"],
                    "domain_adaptation",
                    "Adapted from invoice processing to medical record processing"
                )
        else:
            print("No evolved agents found in the library.")
        
        print("\nEvolution demonstration completed!")
        
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())