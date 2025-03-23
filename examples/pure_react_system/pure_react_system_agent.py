# examples/pure_react_system_agent.py

import asyncio
import logging
import os
import sys
import json
from typing import Dict, Any, List, Optional

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evolving_agents.smart_library.smart_library import SmartLibrary
from evolving_agents.core.llm_service import LLMService
from evolving_agents.core.system_agent import SystemAgentFactory
from evolving_agents.agent_bus.smart_agent_bus import SmartAgentBus

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Sample document data
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

async def main():
    try:
        print("\n" + "="*80)
        print("PURE REACTAGENT SYSTEM AGENT DEMONSTRATION - WITH TOOL ENCAPSULATION")
        print("="*80)
        
        # Initialize components
        library_path = "simplified_agent_library.json"
        
        # Check if library exists, and if not, set up the library first
        if not os.path.exists(library_path):
            print("Library not found. Please run setup_simplified_agent_library.py first.")
            return
        
        # Initialize the Smart Library and LLM service
        library = SmartLibrary(library_path)
        llm_service = LLMService(provider="openai", model="gpt-4o")
        
        # Initialize the Agent Bus
        agent_bus = SmartAgentBus(
            smart_library=library,
            system_agent=None,  # We'll set this after system_agent is created
            storage_path="smart_agent_bus.json", 
            log_path="agent_bus_logs.json"
        )
        
        # Create the SystemAgent as a pure BeeAI ReActAgent
        print("\nInitializing SystemAgent as a pure BeeAI ReActAgent...")
        system_agent = await SystemAgentFactory.create_agent(
            llm_service=llm_service,
            smart_library=library,
            agent_bus=agent_bus,
            memory_type="token"
        )
        
        print("SystemAgent initialized with the following tools:")
        for tool in system_agent.meta.tools:
            print(f"  - {tool.name}: {tool.description}")
            
        # DEMO 1: Show how SearchComponentTool contains decision logic
        print("\n" + "="*80)
        print("DEMO 1: SEARCH COMPONENT TOOL WITH EMBEDDED DECISION LOGIC")
        print("="*80)
        
        search_prompt = """
        Search for components related to document analysis and provide a recommendation
        on whether to reuse, evolve, or create a new component based on the similarity scores.
        """
        
        print("\nPrompt: " + search_prompt)
        print("\nAgent processing...")
        
        # Run the agent - BeeAI's Run object doesn't have .on() method, so we use .result directly
        search_response = await system_agent.run(search_prompt)
        
        print("\nSystemAgent's response:")
        print(search_response.result.text)
        
        # DEMO 2: Show how CreateComponentTool handles creation
        print("\n" + "="*80)
        print("DEMO 2: CREATE COMPONENT TOOL WITH EMBEDDED CREATION STRATEGIES")
        print("="*80)
        
        create_prompt = f"""
        Create a new invoice processing tool using the following requirements:
        
        1. It should be able to extract key information from invoices like vendor, date, items, and total
        2. It should handle different invoice formats 
        3. It should validate that extracted data matches expected patterns
        
        Here's a sample invoice:
        {SAMPLE_INVOICE[:200]}...
        
        Select an appropriate framework and generate code for this tool.
        """
        
        print("\nPrompt: " + create_prompt)
        print("\nAgent processing...")
        
        create_response = await system_agent.run(create_prompt)
        
        print("\nSystemAgent's response:")
        print(create_response.result.text)
        
        # DEMO 3: Show how EvolveComponentTool handles different evolution strategies
        print("\n" + "="*80)
        print("DEMO 3: EVOLVE COMPONENT TOOL WITH MULTIPLE EVOLUTION STRATEGIES")
        print("="*80)
        
        evolve_prompt = """
        Find a document processing tool in our library and evolve it using the 'domain_adaptation' 
        strategy to handle medical records specifically. The evolved tool should be able to:
        
        1. Extract patient information, diagnosis, and treatment plans
        2. Identify key medical terms
        3. Format extracted data in a structured way
        
        Show the different evolution strategies available and explain why domain adaptation 
        is appropriate for this case.
        """
        
        print("\nPrompt: " + evolve_prompt)
        print("\nAgent processing...")
        
        evolve_response = await system_agent.run(evolve_prompt)
        
        print("\nSystemAgent's response:")
        print(evolve_response.result.text)
        
        # DEMO 4: Complete workflow showing all tools working together
        print("\n" + "="*80)
        print("DEMO 4: COMPLETE WORKFLOW - ALL TOOLS WORKING TOGETHER")
        print("="*80)
        
        workflow_prompt = f"""
        I need to process both invoices and medical records. Please:
        
        1. Search for existing components that might help with this task
        2. Based on the search results, determine whether to reuse, evolve, or create components
        3. Create or evolve the necessary components using appropriate strategies
        4. Register the components with the Agent Bus
        5. Process these sample documents:
           - Invoice: {SAMPLE_INVOICE[:100]}...
           - Medical record: {SAMPLE_MEDICAL_RECORD[:100]}...
        
        Explain your decision-making process at each step.
        """
        
        print("\nPrompt: " + workflow_prompt)
        print("\nAgent processing...")
        
        workflow_response = await system_agent.run(workflow_prompt)
        
        print("\nSystemAgent's response:")
        print(workflow_response.result.text)
        
        print("\nDemonstration completed successfully!")
        
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())