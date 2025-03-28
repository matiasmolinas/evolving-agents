# examples/invoice_processing/architect_zero_system_demo.py

import asyncio
import logging
import json
import os
from typing import Dict, Any, List, Optional
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

# Sample invoice for testing
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
        "smart_agent_bus.json",
        "agent_bus_logs.json",
        ".llm_cache",  # Clear the cache to avoid empty responses
        "architect_design.json",
        "invoice_workflow.yaml",
        "sample_invoice.txt",
        "workflow_output.json"
    ]
    
    for file_path in files_to_remove:
        if os.path.exists(file_path):
            try:
                if os.path.isdir(file_path):
                    import shutil
                    shutil.rmtree(file_path)
                else:
                    os.remove(file_path)
                logger.info(f"Removed previous file/directory: {file_path}")
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
    
    # Create dependency container to manage component dependencies
    container = DependencyContainer()
    
    # Step 1: Set up core services
    llm_service = LLMService(provider="openai", model="gpt-4o")
    container.register('llm_service', llm_service)
    
    smart_library = SmartLibrary("smart_library.json", container=container)
    container.register('smart_library', smart_library)
    
    # Create firmware for component creation
    from evolving_agents.firmware.firmware import Firmware
    firmware = Firmware()
    container.register('firmware', firmware)
    
    # Step 2: Create agent bus
    agent_bus = SmartAgentBus(
        storage_path="smart_agent_bus.json",
        log_path="agent_bus_logs.json",
        smart_library=smart_library,
        llm_service=llm_service,
        container=container
    )
    container.register('agent_bus', agent_bus)
    
    # Step 3: Create the system agent
    system_agent = await SystemAgentFactory.create_agent(container=container)
    container.register('system_agent', system_agent)
    
    # Initialize components (but don't try to initialize from library yet)
    await smart_library.initialize()
    
    # Wait until library is initialized before setting up the agent bus
    agent_bus.system_agent = system_agent
    
    # At this point, library exists and agent bus is connected to system agent
    # Let's manually create some agents from the library instead of using initialize_from_library
    await initialize_agents_manually(agent_bus, smart_library)
    
    # Step 4: Create the architect agent using the standard function
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
    """
    
    # Print the task
    logger.info("=== TASK REQUIREMENTS ===")
    logger.info(task_requirement)
    
    try:
        # Step 5: Run the Architect agent with the task
        logger.info("\n=== RUNNING ARCHITECT AGENT ===")
        architect_result = await architect_agent.run(task_requirement)
        
        # Save the architect's design
        design_output = architect_result.result.text if hasattr(architect_result, 'result') else str(architect_result)
        with open("architect_design.json", "w") as f:
            f.write(design_output)
            
        logger.info("Architect design saved to architect_design.json")
        
        # Step 6: Extract and save the workflow YAML if it exists in the result
        yaml_workflow = None
        
        # Try to find a YAML workflow in the result
        if "```yaml" in design_output and "```" in design_output:
            yaml_sections = design_output.split("```yaml")
            for section in yaml_sections[1:]:
                if "```" in section:
                    yaml_workflow = section.split("```")[0].strip()
                    break
        
        # If we found a YAML workflow, save it
        if yaml_workflow:
            with open("invoice_workflow.yaml", "w") as f:
                f.write(yaml_workflow)
                
            logger.info("Workflow YAML extracted and saved to invoice_workflow.yaml")
        else:
            # Create a manual workflow if no valid YAML found
            logger.info("No valid YAML workflow found in the architect's output. Creating a manual workflow.")
            yaml_workflow = create_manual_workflow()
            
            with open("invoice_workflow_manual.yaml", "w") as f:
                f.write(yaml_workflow)
                
            logger.info("Manual workflow saved to invoice_workflow_manual.yaml")
        
        # Step 7: Execute the workflow with the system agent
        if yaml_workflow and system_agent:
            logger.info("\n=== EXECUTING WORKFLOW ===")
            logger.info("Executing the workflow using the System Agent...")
            
            execution_result = await system_agent.workflow_processor.process_workflow(
                yaml_workflow,  # As positional argument 
                params={"invoice_text": SAMPLE_INVOICE}
            )
            
            # Save the execution result
            with open("workflow_output.json", "w") as f:
                json.dump(execution_result, f, indent=2)
                
            logger.info("Workflow execution completed. Result saved to workflow_output.json")
            
            # Display the execution result
            if "result" in execution_result:
                logger.info("\n=== EXECUTION RESULTS ===")
                result = execution_result["result"]
                if isinstance(result, dict):
                    if "summary" in result:
                        logger.info(f"Summary: {result['summary']}")
                    else:
                        logger.info(f"Result: {json.dumps(result, indent=2)}")
                else:
                    logger.info(f"Result: {result}")
        
    except Exception as e:
        logger.error(f"Error in architect process: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())

async def initialize_agents_manually(agent_bus, smart_library):
    """Manually register agents from the library to avoid issues."""
    logger.info("Initializing agents manually from library")
    
    for record in smart_library.records:
        if record.get("status") != "active":
            continue
            
        # Skip if already registered
        agent_name = record["name"]
        if any(a.get("name") == agent_name for a in agent_bus.agents.values()):
            continue
        
        # Create capabilities - make sure we don't use any non-serializable objects
        capabilities = []
        
        # Ensure each capability is a proper dictionary
        if "metadata" in record and "capabilities" in record["metadata"]:
            raw_capabilities = record["metadata"]["capabilities"]
            # Convert any non-dict capabilities to dicts
            for cap in raw_capabilities:
                if isinstance(cap, dict):
                    capabilities.append(cap)
                elif isinstance(cap, str):
                    capabilities.append({
                        "id": cap.lower().replace(" ", "_"),
                        "name": cap,
                        "description": f"Ability to {cap.lower()}",
                        "confidence": 0.8
                    })
                # Skip any non-string, non-dict capabilities
        
        # If no capabilities found, create a default one
        if not capabilities:
            capabilities = [{
                "id": f"{agent_name.lower().replace(' ', '_')}_default",
                "name": agent_name,
                "description": record.get("description", ""),
                "confidence": 0.8
            }]
        
        # Ensure we only have serializable metadata
        metadata = {"source": "SmartLibrary"}
        if "metadata" in record:
            # Copy only serializable fields from record metadata
            for key, value in record["metadata"].items():
                if key != "capabilities" and isinstance(value, (str, int, float, bool, list, dict, type(None))):
                    metadata[key] = value
        
        try:
            # Register the agent
            await agent_bus.register_agent(
                name=agent_name,
                description=record.get("description", ""),
                capabilities=capabilities,
                agent_type=record.get("record_type", "GENERIC"),
                metadata=metadata
            )
            logger.info(f"Registered agent from library: {agent_name}")
        except Exception as e:
            logger.error(f"Error registering agent {agent_name}: {str(e)}")

def create_manual_workflow():
    """Create a manual workflow in case the architect doesn't generate a valid one."""
    return """scenario_name: Advanced Invoice Processing System
domain: document_processing
description: A comprehensive system for processing invoices with verification and summarization

steps:
  # Define our advanced document analyzer
  - type: "DEFINE"
    item_type: "TOOL"
    name: "AdvancedDocumentAnalyzer"
    description: "A sophisticated tool for detecting invoice documents with high confidence"
    code_snippet: |
      from typing import Dict, Any
      from pydantic import BaseModel, Field
      import re

      from beeai_framework.context import RunContext
      from beeai_framework.emitter.emitter import Emitter
      from beeai_framework.tools.tool import StringToolOutput, Tool, ToolRunOptions

      class DocumentAnalyzerInput(BaseModel):
          text: str = Field(description="Document text to analyze")

      class AdvancedDocumentAnalyzer(Tool[DocumentAnalyzerInput, ToolRunOptions, StringToolOutput]):
          \"\"\"A sophisticated tool for detecting document types with high confidence.\"\"\"
          name = "AdvancedDocumentAnalyzer"
          description = "Analyzes document content using multiple heuristics for higher confidence"
          input_schema = DocumentAnalyzerInput

          def _create_emitter(self) -> Emitter:
              return Emitter.root().child(
                  namespace=["tool", "document", "analyzer", "advanced"],
                  creator=self,
              )
          
          async def _run(self, input: DocumentAnalyzerInput, options: ToolRunOptions | None, context: RunContext) -> StringToolOutput:
              \"\"\"Analyze document text to determine its type with high confidence.\"\"\"
              doc_text = input.text.lower()
              
              # Multiple detection features
              features = {
                  "invoice_keywords": ["invoice", "bill", "statement"],
                  "date_patterns": len(re.findall(r'date:|due date:', doc_text, re.IGNORECASE)),
                  "monetary_values": len(re.findall(r'\\$\\d+\\.\\d{2}', doc_text)),
                  "total_indicators": len(re.findall(r'total|subtotal|amount due|balance', doc_text, re.IGNORECASE)),
                  "invoice_number": 1 if re.search(r'invoice #|invoice no|invoice number', doc_text, re.IGNORECASE) else 0,
                  "vendor_info": 1 if re.search(r'vendor|from|seller', doc_text, re.IGNORECASE) else 0,
                  "line_items": len(re.findall(r'\\d+\\..*?\\$\\d+|\\d+\\s+x\\s+', doc_text))
              }
              
              # Calculate confidence for invoice detection
              invoice_indicators = 0
              for keyword in features["invoice_keywords"]:
                  if keyword in doc_text:
                      invoice_indicators += 1
                      
              # Sophisticated confidence calculation
              invoice_confidence = 0.0
              if invoice_indicators > 0:
                  base_confidence = 0.5 + (invoice_indicators * 0.1)  # 0.6-0.8 based on keywords
                  feature_score = (
                      min(features["date_patterns"], 2) * 0.05 +
                      min(features["monetary_values"], 5) * 0.02 +
                      min(features["total_indicators"], 3) * 0.05 +
                      features["invoice_number"] * 0.1 +
                      features["vendor_info"] * 0.05 +
                      min(features["line_items"], 3) * 0.03
                  )
                  invoice_confidence = min(0.95, base_confidence + feature_score)
              
              # Determine document type
              doc_type = "unknown"
              if invoice_confidence > 0.7:
                  doc_type = "invoice"
              elif "receipt" in doc_text and invoice_confidence > 0.4:
                  doc_type = "receipt"
                  invoice_confidence = min(0.9, invoice_confidence + 0.1)
              
              result = {
                  "type": doc_type,
                  "confidence": round(invoice_confidence, 2),
                  "features": features
              }
              
              import json
              return StringToolOutput(json.dumps(result, indent=2))

  # Define an advanced invoice processor 
  - type: "DEFINE"
    item_type: "AGENT"
    name: "AdvancedInvoiceProcessor"
    description: "An advanced agent that processes invoices with comprehensive information extraction and verification"
    code_snippet: |
      from typing import List, Dict, Any, Optional
      import re
      import json

      from beeai_framework.agents.react import ReActAgent
      from beeai_framework.agents.types import AgentMeta
      from beeai_framework.memory import TokenMemory
      from beeai_framework.backend.chat import ChatModel
      from beeai_framework.tools.tool import Tool

      class AdvancedInvoiceProcessorInitializer:
          \"\"\"
          Advanced invoice processing agent that extracts comprehensive information,
          verifies calculations, and generates structured insights.
          \"\"\"
          
          @staticmethod
          def create_agent(llm: ChatModel, tools: Optional[List[Tool]] = None) -> ReActAgent:
              \"\"\"Create and configure the advanced invoice processor agent.\"\"\"
              if tools is None:
                  tools = []
                  
              meta = AgentMeta(
                  name="AdvancedInvoiceProcessor",
                  description=(
                      "I am an advanced invoice processing agent that extracts comprehensive information, "
                      "verifies calculations, and generates insights."
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
          async def process_invoice(invoice_text: str) -> Dict[str, Any]:
              \"\"\"
              Process an invoice to extract comprehensive information.
              
              Args:
                  invoice_text: The text of the invoice to process
                  
              Returns:
                  Extracted invoice information with verification
              \"\"\"
              # Extract core information
              invoice_num_match = re.search(r'INVOICE\\s*#?([\\w-]+)', invoice_text, re.IGNORECASE)
              invoice_num = invoice_num_match.group(1) if invoice_num_match else "Unknown"
              
              date_match = re.search(r'Date:?\\s*([\\w\\d/-]+)', invoice_text, re.IGNORECASE)
              date = date_match.group(1).strip() if date_match else "Unknown"
              
              vendor_match = re.search(r'Vendor:?\\s*([^\\n]+)', invoice_text, re.IGNORECASE)
              vendor = vendor_match.group(1).strip() if vendor_match else "Unknown"
              
              # Extract billing information
              bill_to_match = re.search(r'Bill\\s*To:([^\\n]+(?:\\n[^\\n]+)*?)\\n\\s*\\n', invoice_text, re.IGNORECASE)
              bill_to = bill_to_match.group(1).strip() if bill_to_match else "Unknown"
              
              # Extract financial information
              subtotal_match = re.search(r'Subtotal:?\\s*\\$?([\\d,.]+)', invoice_text, re.IGNORECASE)
              subtotal_str = subtotal_match.group(1).strip() if subtotal_match else "0.00"
              subtotal = float(subtotal_str.replace(',', ''))
              
              tax_match = re.search(r'Tax\\s*(?:\\([^)]*\\))?:?\\s*\\$?([\\d,.]+)', invoice_text, re.IGNORECASE)
              tax_str = tax_match.group(1).strip() if tax_match else "0.00"
              tax = float(tax_str.replace(',', ''))
              
              total_match = re.search(r'Total\\s*(?:Due|Amount)?:?\\s*\\$?([\\d,.]+)', invoice_text, re.IGNORECASE)
              total_str = total_match.group(1).strip() if total_match else "0.00"
              total = float(total_str.replace(',', ''))
              
              # Extract line items using regex pattern matching
              items_section_match = re.search(r'Items?:?([^\\n]*(?:\\n[^\\n]+)*?)\\n\\s*(?:Subtotal|Total|Tax)', invoice_text, re.IGNORECASE)
              items_text = items_section_match.group(1) if items_section_match else ""
              
              # Parse individual line items
              items = []
              for line in items_text.split('\\n'):
                  if not line.strip():
                      continue
                      
                  # Try different patterns for line items
                  item_match = re.search(r'(\d+)\.\s+([^-$]+)\s*-\s*\$?([0-9,.]+)(?:\s*\((\d+)\s*units?\))?', line)
                  if item_match:
                      name = item_match.group(2).strip()
                      unit_price = float(item_match.group(3).replace(',', ''))
                      quantity = int(item_match.group(4)) if item_match.group(4) else 1
                      item_total = unit_price * quantity
                      
                      items.append({
                          "name": name,
                          "unit_price": unit_price,
                          "quantity": quantity,
                          "total": item_total
                      })
              
              # Calculate expected total
              calculated_total = subtotal + tax
              total_matches = abs(calculated_total - total) < 0.01
              
              # Calculate subtotal from line items
              items_total = sum(item["total"] for item in items)
              subtotal_matches = abs(items_total - subtotal) < 0.01
              
              # Create verification section
              verification = {
                  "total_calculation_correct": total_matches,
                  "subtotal_matches_line_items": subtotal_matches,
                  "discrepancies": []
              }
              
              if not total_matches:
                  verification["discrepancies"].append(f"Total (${total:.2f}) doesn't match Subtotal + Tax (${calculated_total:.2f})")
                  
              if not subtotal_matches and items:
                  verification["discrepancies"].append(f"Subtotal (${subtotal:.2f}) doesn't match sum of line items (${items_total:.2f})")
              
              # Create summary
              summary = f"Invoice #{invoice_num} from {vendor} dated {date} for ${total:.2f}"
              if verification["discrepancies"]:
                  summary += " has calculation discrepancies that should be reviewed."
              else:
                  summary += " has been verified with all calculations correct."
              
              return {
                  "invoice_number": invoice_num,
                  "date": date,
                  "vendor": vendor,
                  "bill_to": bill_to,
                  "items": items,
                  "subtotal": subtotal,
                  "tax": tax,
                  "total": total,
                  "verification": verification,
                  "summary": summary
              }

  # Execute the workflow
  - type: "EXECUTE"
    item_type: "TOOL"
    name: "AdvancedDocumentAnalyzer"
    input:
      text: "{{params.invoice_text}}"
    output_var: "document_analysis"
    
  - type: "EXECUTE"
    item_type: "AGENT"
    name: "AdvancedInvoiceProcessor"
    method: "process_invoice"
    input: "{{params.invoice_text}}"
    output_var: "invoice_data"
    condition: "document_analysis.type == 'invoice'"
    
  - type: "RETURN"
    value: "{{invoice_data}}"
"""

if __name__ == "__main__":
    asyncio.run(main())