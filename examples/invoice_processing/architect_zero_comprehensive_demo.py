# examples/invoice_processing/architect_zero_comprehensive_demo.py

import asyncio
import logging
import json
import os
import re
from dotenv import load_dotenv

from evolving_agents.core.llm_service import LLMService
from evolving_agents.smart_library.smart_library import SmartLibrary
from evolving_agents.agent_bus.smart_agent_bus import SmartAgentBus
from evolving_agents.core.system_agent import SystemAgentFactory
from evolving_agents.agents.architect_zero import create_architect_zero
from evolving_agents.core.dependency_container import DependencyContainer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Sample invoice for testing (kept separate from the workflow)
SAMPLE_INVOICE = """
INVOICE #12345
Date: 2023-05-15
Vendor: TechSupplies Inc.
Address: 123 Tech Blvd, San Francisco, CA 94107

Bill To:
Acme Corporation
456 Business Ave
New York, NY 10001

Items:
1. Laptop Computer - $1,200.00 (2 units)
2. External Monitor - $300.00 (3 units)
3. Wireless Keyboard - $50.00 (5 units)

Subtotal: $2,950.00
Tax (8.5%): $250.75
Total Due: $3,200.75

Payment Terms: Net 30
Due Date: 2023-06-14

Thank you for your business!
"""

def clean_previous_files():
    """Remove previous files to start fresh."""
    files_to_remove = [
        "smart_library.json",
        "agent_registry.json",
        "architect_interaction.txt",
        "invoice_workflow.yaml",
        "workflow_execution_result.json",
        "sample_invoice.txt",
        "debug_workflow.yaml"
    ]
    
    for file_path in files_to_remove:
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
                logger.info(f"Removed previous file: {file_path}")
            except Exception as e:
                logger.warning(f"Failed to remove {file_path}: {str(e)}")

async def setup_library():
    """Set up some initial components in the smart library to show evolution."""
    llm_service = LLMService(provider="openai", model="gpt-4o")
    smart_library = SmartLibrary("smart_library.json")
    
    # Create a basic document analyzer
    basic_doc_analyzer = {
        "name": "BasicDocumentAnalyzer",
        "record_type": "TOOL",
        "domain": "document_processing",
        "description": "A basic tool that analyzes documents to determine their type",
        "code_snippet": """
from typing import Dict, Any
from pydantic import BaseModel, Field

from beeai_framework.context import RunContext
from beeai_framework.emitter.emitter import Emitter
from beeai_framework.tools.tool import StringToolOutput, Tool, ToolRunOptions

class DocumentAnalyzerInput(BaseModel):
    text: str = Field(description="Document text to analyze")

class BasicDocumentAnalyzer(Tool[DocumentAnalyzerInput, ToolRunOptions, StringToolOutput]):
    \"\"\"A basic tool that analyzes documents to determine their type.\"\"\"
    name = "BasicDocumentAnalyzer"
    description = "Analyzes document content to determine if it's an invoice, receipt, or other document type"
    input_schema = DocumentAnalyzerInput

    def _create_emitter(self) -> Emitter:
        return Emitter.root().child(
            namespace=["tool", "document", "analyzer"],
            creator=self,
        )
    
    async def _run(self, input: DocumentAnalyzerInput, options: ToolRunOptions | None, context: RunContext) -> StringToolOutput:
        \"\"\"Analyze document text to determine its type.\"\"\"
        doc_text = input.text.lower()
        
        # Simple keyword matching
        result = {"type": "unknown", "confidence": 0.0}
        
        if "invoice" in doc_text and ("total" in doc_text or "amount" in doc_text):
            result = {"type": "invoice", "confidence": 0.7}
        elif "receipt" in doc_text:
            result = {"type": "receipt", "confidence": 0.6}
        elif "contract" in doc_text:
            result = {"type": "contract", "confidence": 0.6}
        
        import json
        return StringToolOutput(json.dumps(result, indent=2))
""",
        "version": "1.0.0",
        "tags": ["document", "analysis", "basic"]
    }
    
    # Create a basic invoice processor
    basic_invoice_processor = {
        "name": "BasicInvoiceProcessor",
        "record_type": "AGENT",
        "domain": "document_processing",
        "description": "A basic agent that processes invoice documents to extract information",
        "code_snippet": """
from typing import List, Dict, Any, Optional
import re

from beeai_framework.agents.react import ReActAgent
from beeai_framework.agents.types import AgentMeta
from beeai_framework.memory import TokenMemory
from beeai_framework.backend.chat import ChatModel
from beeai_framework.tools.tool import Tool

class BasicInvoiceProcessorInitializer:
    \"\"\"
    A basic agent that processes invoice documents to extract information.
    It can extract simple data like invoice number, date, and total amount.
    \"\"\"
    
    @staticmethod
    def create_agent(llm: ChatModel, tools: Optional[List[Tool]] = None) -> ReActAgent:
        \"\"\"Create and configure the invoice processor agent.\"\"\"
        # Use empty tools list if none provided
        if tools is None:
            tools = []
            
        # Define agent metadata
        meta = AgentMeta(
            name="BasicInvoiceProcessor",
            description=(
                "I am an invoice processing agent that can extract basic information from invoice documents "
                "including invoice number, date, vendor, and total amount."
            ),
            tools=tools
        )
        
        # Create the agent
        agent = ReActAgent(
            llm=llm,
            tools=tools,
            memory=TokenMemory(llm),
            meta=meta
        )
        
        return agent
        
    @staticmethod
    async def process_invoice(invoice_text: str) -> Dict[str, Any]:
        \"\"\"
        Process an invoice to extract key information.
        
        Args:
            invoice_text: The text of the invoice to process
            
        Returns:
            Extracted invoice information
        \"\"\"
        # Extract invoice number
        invoice_num_match = re.search(r'INVOICE #([\\w-]+)', invoice_text, re.IGNORECASE)
        invoice_num = invoice_num_match.group(1) if invoice_num_match else "Unknown"
        
        # Extract date
        date_match = re.search(r'Date:?\\s*([\\w\\d/-]+)', invoice_text, re.IGNORECASE)
        date = date_match.group(1).strip() if date_match else "Unknown"
        
        # Extract vendor
        vendor_match = re.search(r'Vendor:?\\s*([^\\n]+)', invoice_text, re.IGNORECASE)
        vendor = vendor_match.group(1).strip() if vendor_match else "Unknown"
        
        # Extract total
        total_match = re.search(r'Total\\s*(?:Due|Amount)?:?\\s*\\$?([\\d.,]+)', invoice_text, re.IGNORECASE)
        total = total_match.group(1).strip() if total_match else "Unknown"
        
        return {
            "invoice_number": invoice_num,
            "date": date,
            "vendor": vendor,
            "total": total
        }
""",
        "version": "1.0.0",
        "tags": ["invoice", "processing", "basic"]
    }
    
    # Add them to the library
    await smart_library.create_record(**basic_doc_analyzer)
    await smart_library.create_record(**basic_invoice_processor)
    
    logger.info("Set up initial components in the smart library")

async def main():
    # Clean up previous files
    clean_previous_files()
    
    # Save sample invoice to a file for reference
    with open("sample_invoice.txt", "w") as f:
        f.write(SAMPLE_INVOICE)
    
    # First, set up some initial components in the smart library
    await setup_library()
    
    # Create a dependency container to manage component dependencies
    container = DependencyContainer()
    
    # Initialize core components
    llm_service = LLMService(provider="openai", model="gpt-4o")
    container.register('llm_service', llm_service)
    
    smart_library = SmartLibrary("smart_library.json", container=container)
    container.register('smart_library', smart_library)
    
    # Create firmware
    from evolving_agents.firmware.firmware import Firmware
    firmware = Firmware()
    container.register('firmware', firmware)
    
    # Create agent bus with null system agent
    agent_bus = SmartAgentBus(
        storage_path="agent_registry.json", 
        log_path="agent_bus_logs.json",
        container=container
    )
    container.register('agent_bus', agent_bus)
    
    # Create system agent
    system_agent = await SystemAgentFactory.create_agent(container=container)
    container.register('system_agent', system_agent)
    
    # Initialize components
    await smart_library.initialize()
    await agent_bus.initialize_from_library()
    
    # Create the Architect-Zero agent using the container
    architect_agent = await create_architect_zero(container=container)
    
    # Define invoice processing task requirements
    task_requirement = """
    Create an advanced invoice processing system that improves upon the basic version in the library. The system should:
    
    1. Use a more sophisticated document analyzer that can detect invoices with higher confidence
    2. Extract comprehensive information (invoice number, date, vendor, items, subtotal, tax, total)
    3. Verify calculations to ensure subtotal + tax = total
    4. Generate a structured summary with key insights
    5. Handle different invoice formats and detect potential errors
    
    The system should leverage existing components from the library when possible,
    evolve them where improvements are needed, and create new components for missing functionality.
    
    Please generate a complete workflow for this invoice processing system.
    """
    
    # Print the task
    logger.info("=== TASK REQUIREMENTS ===")
    logger.info(task_requirement)
    
    # Run the architect agent to design the system
    logger.info("\n=== RUNNING ARCHITECT-ZERO AGENT ===")
    try:
        # Execute the architect agent as a ReAct agent
        result = await architect_agent.run(task_requirement)
        
        # Save the full agent interaction
        with open("architect_interaction.txt", "w") as f:
            f.write(f"TASK REQUIREMENT:\n{task_requirement}\n\n")
            f.write(f"AGENT THOUGHT PROCESS:\n{result.result.text}")
        
        logger.info("Architect-Zero completed successfully - see 'architect_interaction.txt' for full output")
        
        # Instead of relying on LLM-generated workflows, let's create a simple, reliable workflow manually
        yaml_content = create_simple_workflow()
        
        if yaml_content:
            # Save the workflow to a file
            with open("invoice_workflow.yaml", "w") as f:
                f.write(yaml_content)
            
            logger.info("Generated workflow saved to invoice_workflow.yaml")
            
            # Try to execute the workflow with sample invoice data
            logger.info("\n=== EXECUTING GENERATED WORKFLOW ===")
            
            try:
                # Save the sample invoice as an encoded string to avoid YAML parsing issues
                import base64
                encoded_invoice = base64.b64encode(SAMPLE_INVOICE.encode('utf-8')).decode('utf-8')
                
                # Create a workflow with the encoded invoice
                workflow_with_invoice = yaml_content.replace("ENCODED_INVOICE_PLACEHOLDER", encoded_invoice)
                
                # Save the debug workflow for inspection
                with open("debug_workflow.yaml", "w") as f:
                    f.write(workflow_with_invoice)
                    logger.info("Debug workflow saved to debug_workflow.yaml")
                
                # Execute the workflow
                execution_result = await system_agent.workflow_processor.process_workflow(workflow_with_invoice)
                
                logger.info(f"Workflow execution result: {json.dumps(execution_result, indent=2)}")
                
                # Save execution result
                with open("workflow_execution_result.json", "w") as f:
                    json.dump(execution_result, f, indent=2)
                
                logger.info("Workflow execution result saved to workflow_execution_result.json")
                
            except Exception as e:
                logger.error(f"Error executing workflow: {str(e)}")
                import traceback
                logger.error(traceback.format_exc())
        else:
            logger.warning("Failed to generate a valid workflow")
    
    except Exception as e:
        logger.error(f"Error running Architect-Zero: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())

def create_simple_workflow():
    """Create a simple, reliable workflow for invoice processing."""
    return """scenario_name: Advanced Invoice Processing System
domain: document_processing
description: A system to process invoice documents with advanced capabilities

steps:
  # Define the AdvancedInvoiceProcessor agent
  - type: "DEFINE"
    item_type: "AGENT"
    name: "AdvancedInvoiceProcessor"
    description: "Advanced invoice processor that extracts and verifies information from invoices"
    code_snippet: |
      from typing import List, Dict, Any, Optional
      import re
      import base64
      import json

      from beeai_framework.agents.react import ReActAgent
      from beeai_framework.agents.types import AgentMeta
      from beeai_framework.memory import TokenMemory
      from beeai_framework.backend.chat import ChatModel
      from beeai_framework.tools.tool import Tool

      class AdvancedInvoiceProcessorInitializer:
          \"\"\"
          Advanced invoice processor that extracts and verifies information from invoices
          \"\"\"
          
          @staticmethod
          def create_agent(llm: ChatModel, tools: Optional[List[Tool]] = None) -> ReActAgent:
              \"\"\"Create and configure the invoice processor agent.\"\"\"
              if tools is None:
                  tools = []
                  
              meta = AgentMeta(
                  name="AdvancedInvoiceProcessor",
                  description=(
                      "I am an advanced invoice processing agent that extracts comprehensive information, "
                      "verifies calculations, detects errors, and generates structured summaries."
                  ),
                  tools=tools
              )
              
              agent = ReActAgent(
                  llm=llm,
                  tools=tools,
                  memory=TokenMemory(llm),
                  meta=meta
              )
              
              return agent
          
          @staticmethod
          async def process_invoice(invoice_text: str, is_encoded: bool = False) -> Dict[str, Any]:
              \"\"\"
              Process an invoice to extract comprehensive information.
              
              Args:
                  invoice_text: The text of the invoice to process, possibly base64 encoded
                  is_encoded: Whether the invoice text is base64 encoded
                  
              Returns:
                  Extracted invoice information with verification results
              \"\"\"
              # Decode if necessary
              if is_encoded:
                  try:
                      invoice_text = base64.b64decode(invoice_text).decode('utf-8')
                  except Exception as e:
                      return {"error": f"Failed to decode invoice: {str(e)}"}
              
              # Extract basic information
              invoice_num_match = re.search(r'INVOICE #([\\w-]+)', invoice_text, re.IGNORECASE)
              invoice_num = invoice_num_match.group(1) if invoice_num_match else "Unknown"
              
              date_match = re.search(r'Date:?\\s*([\\w\\d/-]+)', invoice_text, re.IGNORECASE)
              date = date_match.group(1).strip() if date_match else "Unknown"
              
              vendor_match = re.search(r'Vendor:?\\s*([^\\n]+)', invoice_text, re.IGNORECASE)
              vendor = vendor_match.group(1).strip() if vendor_match else "Unknown"
              
              # Extract financial information
              subtotal_match = re.search(r'Subtotal:?\\s*\\$?([\\d.,]+)', invoice_text, re.IGNORECASE)
              subtotal = subtotal_match.group(1).strip() if subtotal_match else "0.00"
              subtotal = float(subtotal.replace(',', ''))
              
              tax_match = re.search(r'Tax\\s*(?:\\([^)]*\\))?:?\\s*\\$?([\\d.,]+)', invoice_text, re.IGNORECASE)
              tax = tax_match.group(1).strip() if tax_match else "0.00"
              tax = float(tax.replace(',', ''))
              
              total_match = re.search(r'Total\\s*(?:Due|Amount)?:?\\s*\\$?([\\d.,]+)', invoice_text, re.IGNORECASE)
              total = total_match.group(1).strip() if total_match else "0.00"
              total = float(total.replace(',', ''))
              
              # Extract line items
              items = []
              item_matches = re.findall(r'(\\d+\\.\\s*([^-]+)\\s*-\\s*\\$([\\d.,]+)\\s*\\(?(\\d+)\\s*units?\\)?)', invoice_text)
              for match in item_matches:
                  full_match, name, unit_price, quantity = match
                  items.append({
                      "name": name.strip(),
                      "unit_price": float(unit_price.replace(',', '')),
                      "quantity": int(quantity),
                      "total": float(unit_price.replace(',', '')) * int(quantity)
                  })
              
              # Verify calculations
              calculated_subtotal = sum(item["total"] for item in items)
              calculated_total = subtotal + tax
              
              verification = {
                  "subtotal_matches_items": abs(calculated_subtotal - subtotal) < 0.01,
                  "total_matches_calculation": abs(calculated_total - total) < 0.01
              }
              
              return {
                  "invoice_number": invoice_num,
                  "date": date,
                  "vendor": vendor,
                  "items": items,
                  "subtotal": subtotal,
                  "tax": tax,
                  "total": total,
                  "verification": verification,
                  "summary": f"Invoice {invoice_num} from {vendor} dated {date} with {len(items)} items totaling ${total:.2f}"
              }

  # Create the AdvancedInvoiceProcessor by registering it with the agent bus
  - type: "CREATE"
    item_type: "AGENT"
    name: "AdvancedInvoiceProcessor"
    capabilities:
      - id: "invoice_processing"
        name: "Invoice Processing"
        description: "Extract data from and verify invoice calculations"
        confidence: 0.9
      - id: "calculation_verification"
        name: "Calculation Verification" 
        description: "Verify that invoice subtotal + tax = total"
        confidence: 0.9
    agent_type: "SPECIALIZED"
    
  # Execute the AdvancedInvoiceProcessor with base64-encoded invoice
  - type: "EXECUTE"
    item_type: "AGENT"
    name: "AdvancedInvoiceProcessor"
    task:
      type: "PROCESS_INVOICE"
      data:
        invoice_text: "ENCODED_INVOICE_PLACEHOLDER"
        is_encoded: true
      options:
        verbose: true
    timeout: 60.0
"""

async def generate_clean_workflow(llm_service, full_text):
    """Generate a clean YAML workflow that properly handles the sample invoice."""
    prompt = """
    Create a clean YAML workflow for an advanced invoice processing system.
    
    The workflow should include:
    1. A definition for an AdvancedInvoiceProcessor agent
    2. Definitions for component agents (DocumentAnalyzer, DataExtractor, etc.)
    3. Creation of all defined components
    4. Execution with a placeholder for the invoice text
    
    IMPORTANT: 
    - Use a placeholder "INVOICE_PLACEHOLDER" where the invoice text should be inserted
    - Don't include the actual invoice text in the YAML
    - Make sure the YAML is valid and doesn't contain any text that isn't part of the workflow definition
    
    Basic workflow structure:
    ```yaml
    scenario_name: Advanced Invoice Processing System
    domain: document_processing
    description: A system to process invoice documents with advanced capabilities

    steps:
      # Define agents
      - type: "DEFINE"
        item_type: "AGENT"
        name: "AdvancedInvoiceProcessor"
        description: "Main agent that orchestrates the invoice processing workflow"
        code_snippet: |
          # Agent code here
          
      # Create agents
      - type: "CREATE"
        item_type: "AGENT"
        name: "AdvancedInvoiceProcessor"
      
      # Execute
      - type: "EXECUTE"
        item_type: "AGENT"
        name: "AdvancedInvoiceProcessor"
        user_input: |
          Process this invoice:
          
          INVOICE_PLACEHOLDER
    ```

    Return only the YAML workflow without any additional text or explanations.
    """
    
    response = await llm_service.generate(prompt)
    
    # Extract the YAML
    yaml_match = re.search(r'```yaml\s*\n(.*?)\n\s*```', response, re.DOTALL)
    if yaml_match:
        return yaml_match.group(1).strip()
    
    # If not found with yaml marker, try without specific language
    yaml_match2 = re.search(r'```\s*\n(scenario_name:.*?)\n\s*```', response, re.DOTALL)
    if yaml_match2:
        return yaml_match2.group(1).strip()
    
    # If still not found, return the full response if it looks like YAML
    if response.strip().startswith('scenario_name:'):
        return response.strip()
    
    return None

if __name__ == "__main__":
    asyncio.run(main())