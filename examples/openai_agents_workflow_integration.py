# examples/complete_openai_agents_integration.py

import asyncio
import logging
import os
import sys
import yaml
import json
from typing import Dict, Any

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evolving_agents.smart_library.smart_library import SmartLibrary
from evolving_agents.core.llm_service import LLMService
from evolving_agents.core.system_agent import SystemAgentFactory
from evolving_agents.agent_bus.smart_agent_bus import SmartAgentBus
from evolving_agents.workflow.workflow_generator import WorkflowGenerator
from evolving_agents.workflow.workflow_processor import WorkflowProcessor
from evolving_agents.providers.registry import ProviderRegistry
from evolving_agents.providers.beeai_provider import BeeAIProvider
from evolving_agents.providers.openai_agents_provider import OpenAIAgentsProvider
from evolving_agents.tools.tool_factory import ToolFactory
from evolving_agents.agents.agent_factory import AgentFactory

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Sample document data for demonstration
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

# Document analyzer tool code snippet
DOCUMENT_ANALYZER_CODE = '''
from pydantic import BaseModel, Field
import json
import re

from beeai_framework.context import RunContext
from beeai_framework.emitter.emitter import Emitter
from beeai_framework.tools.tool import StringToolOutput, Tool, ToolRunOptions

class DocumentAnalyzerInput(BaseModel):
    text: str = Field(description="Document text to analyze")

class DocumentAnalyzer(Tool[DocumentAnalyzerInput, ToolRunOptions, StringToolOutput]):
    """Tool that analyzes and identifies the type of document from its content."""
    name = "DocumentAnalyzer"
    description = "Analyzes document content to determine its type and key characteristics"
    input_schema = DocumentAnalyzerInput

    def _create_emitter(self) -> Emitter:
        return Emitter.root().child(
            namespace=["tool", "document", "analyzer"],
            creator=self,
        )
    
    async def _run(self, input: DocumentAnalyzerInput, options: ToolRunOptions | None, context: RunContext) -> StringToolOutput:
        """Analyze a document to identify its type based on content patterns."""
        document_text = input.text.lower()
        
        # Define type indicators with confidence scores
        type_indicators = {
            "invoice": ["invoice", "bill", "payment", "due date", "total due", "vendor", "subtotal", "tax"],
            "receipt": ["receipt", "payment received", "thank you for your purchase", "change", "cashier"],
            "medical_record": ["patient", "diagnosis", "treatment", "prescription", "doctor", "hospital", "medical"],
            "contract": ["agreement", "terms and conditions", "parties", "signed", "effective date", "termination"],
            "resume": ["experience", "education", "skills", "employment", "resume", "cv", "curriculum vitae"],
            "letter": ["dear", "sincerely", "regards", "to whom it may concern"],
            "report": ["report", "findings", "analysis", "conclusion", "executive summary"],
        }
        
        # Calculate confidence scores for each document type
        scores = {}
        for doc_type, indicators in type_indicators.items():
            score = 0
            for indicator in indicators:
                if indicator in document_text:
                    score += 1
            if score > 0:
                confidence = min(0.9, score / len(indicators))  # Cap at 90% confidence
                scores[doc_type] = confidence
        
        # Determine the most likely document type
        if scores:
            most_likely_type = max(scores.items(), key=lambda x: x[1])
            doc_type, confidence = most_likely_type
        else:
            doc_type = "unknown"
            confidence = 0.0
        
        # Extract key patterns based on document type
        extracted_fields = {}
        
        # For invoices, get some key fields
        if doc_type == "invoice":
            # Extract invoice number
            invoice_num_match = re.search(r'(?:invoice|inv)\\s*[#:]?\\s*(\\w+)', document_text, re.IGNORECASE)
            if invoice_num_match:
                extracted_fields["invoice_number"] = invoice_num_match.group(1)
                
            # Extract total
            total_match = re.search(r'total\\s*(?:due|amount)?\\s*[:$]?\\s*(\\d+[.,]?\\d*)', document_text)
            if total_match:
                extracted_fields["total"] = total_match.group(1)
        
        # Build the response
        result = {
            "document_type": doc_type,
            "confidence": confidence,
            "possible_types": [k for k, v in scores.items() if v > 0.2],
            "extracted_fields": extracted_fields
        }
        
        return StringToolOutput(json.dumps(result, indent=2))
'''

# Calculation verifier tool code snippet
CALCULATION_VERIFIER_CODE = '''
from pydantic import BaseModel, Field
import json
import re

from beeai_framework.context import RunContext
from beeai_framework.emitter.emitter import Emitter
from beeai_framework.tools.tool import StringToolOutput, Tool, ToolRunOptions

class CalculationVerifierInput(BaseModel):
    invoice_data: str = Field(description="JSON string containing invoice data with subtotal, tax, and total fields")

class CalculationVerifier(Tool[CalculationVerifierInput, ToolRunOptions, StringToolOutput]):
    """Tool that verifies calculations in invoices, ensuring subtotal + tax = total."""
    name = "CalculationVerifier"
    description = "Verifies that calculations in an invoice are correct (subtotal + tax = total)"
    input_schema = CalculationVerifierInput

    def _create_emitter(self) -> Emitter:
        return Emitter.root().child(
            namespace=["tool", "invoice", "verifier"],
            creator=self,
        )
    
    async def _run(self, input: CalculationVerifierInput, options: ToolRunOptions | None, context: RunContext) -> StringToolOutput:
        """Verify that calculations in an invoice are correct (subtotal + tax = total)."""
        try:
            # Parse the invoice data
            data = json.loads(input.invoice_data) if isinstance(input.invoice_data, str) else input.invoice_data
            
            # Extract the values
            subtotal = None
            tax = None
            total = None
            
            if "subtotal" in data:
                subtotal = float(str(data["subtotal"]).replace("$", "").replace(",", ""))
            if "tax" in data:
                if isinstance(data["tax"], dict) and "amount" in data["tax"]:
                    tax = float(str(data["tax"]["amount"]).replace("$", "").replace(",", ""))
                else:
                    tax = float(str(data["tax"]).replace("$", "").replace(",", ""))
            if "total_due" in data:
                total = float(str(data["total_due"]).replace("$", "").replace(",", ""))
            elif "total" in data:
                total = float(str(data["total"]).replace("$", "").replace(",", ""))
            
            # Verify the calculation
            if subtotal is not None and tax is not None and total is not None:
                expected_total = subtotal + tax
                is_correct = abs(expected_total - total) < 0.01  # Allow for small rounding differences
                
                result = {
                    "is_correct": is_correct,
                    "subtotal": subtotal,
                    "tax": tax,
                    "expected_total": expected_total,
                    "actual_total": total,
                    "difference": total - expected_total
                }
                
                return StringToolOutput(json.dumps(result, indent=2))
            else:
                return StringToolOutput(json.dumps({
                    "is_correct": False,
                    "error": "Missing required fields (subtotal, tax, or total)",
                    "available_fields": list(data.keys())
                }, indent=2))
                
        except Exception as e:
            return StringToolOutput(json.dumps({
                "is_correct": False,
                "error": f"Error verifying calculations: {str(e)}"
            }, indent=2))
'''

# Invoice summary generator tool code snippet
INVOICE_SUMMARY_GENERATOR_CODE = '''
from pydantic import BaseModel, Field
import json
from datetime import datetime

from beeai_framework.context import RunContext
from beeai_framework.emitter.emitter import Emitter
from beeai_framework.tools.tool import StringToolOutput, Tool, ToolRunOptions

class InvoiceSummaryInput(BaseModel):
    invoice_data: str = Field(description="JSON string containing structured invoice data")

class InvoiceSummaryGenerator(Tool[InvoiceSummaryInput, ToolRunOptions, StringToolOutput]):
    """Tool that generates a concise summary of an invoice's key information."""
    name = "InvoiceSummaryGenerator"
    description = "Generates a concise summary from structured invoice data"
    input_schema = InvoiceSummaryInput

    def _create_emitter(self) -> Emitter:
        return Emitter.root().child(
            namespace=["tool", "invoice", "summary"],
            creator=self,
        )
    
    async def _run(self, input: InvoiceSummaryInput, options: ToolRunOptions | None, context: RunContext) -> StringToolOutput:
        """Generate a concise summary of an invoice from structured data."""
        try:
            # Parse the invoice data
            data = json.loads(input.invoice_data) if isinstance(input.invoice_data, str) else input.invoice_data
            
            # Extract key information
            invoice_number = data.get("invoice_number", "Unknown")
            date = data.get("date", "Unknown date")
            vendor_name = data.get("vendor", {}).get("name", data.get("vendor", "Unknown vendor"))
            total_due = data.get("total_due", data.get("total", "Unknown amount"))
            due_date = data.get("due_date", "Unknown")
            
            # Format as currency if needed
            if isinstance(total_due, (int, float)):
                total_due = f"${total_due:,.2f}"
            
            # Generate the summary
            summary = {
                "summary": f"Invoice #{invoice_number} from {vendor_name}",
                "key_details": {
                    "invoice_number": invoice_number,
                    "vendor": vendor_name,
                    "date": date,
                    "total_due": total_due,
                    "due_date": due_date
                },
                "line_item_count": len(data.get("items", [])),
                "recommendations": []
            }
            
            # Add recommendations based on the data
            if due_date:
                try:
                    due_date_obj = datetime.strptime(due_date, "%Y-%m-%d")
                    today = datetime.now()
                    days_until_due = (due_date_obj - today).days
                    
                    if days_until_due < 0:
                        summary["recommendations"].append("OVERDUE: Payment is past due")
                    elif days_until_due < 7:
                        summary["recommendations"].append(f"URGENT: Payment due soon ({days_until_due} days)")
                    elif days_until_due < 30:
                        summary["recommendations"].append(f"REMINDER: Payment due in {days_until_due} days")
                except:
                    pass
            
            # Check for large amounts
            try:
                amount = float(str(total_due).replace("$", "").replace(",", ""))
                if amount > 1000:
                    summary["recommendations"].append("ATTENTION: Invoice amount exceeds $1,000")
            except:
                pass
                
            return StringToolOutput(json.dumps(summary, indent=2))
            
        except Exception as e:
            return StringToolOutput(json.dumps({
                "error": f"Error generating summary: {str(e)}",
                "partial_summary": "Unable to generate complete summary due to an error"
            }, indent=2))
'''

# OpenAI invoice processor agent code snippet
OPENAI_INVOICE_PROCESSOR_CODE = '''
from agents import Agent, Runner, ModelSettings

# Create an OpenAI agent for invoice processing
agent = Agent(
    name="InvoiceProcessor",
    instructions="""
You are an expert invoice processing agent specialized in financial document analysis.
Extract all important information from invoices including:
- Invoice number
- Date
- Vendor information
- Line items (description, quantity, price)
- Subtotal, taxes, and total
- Payment terms

Verify that calculations are correct.
Flag any suspicious or unusual patterns.
Structure your output clearly and follow all governance rules for financial documents.
""",
    model="gpt-4o",
    model_settings=ModelSettings(
        temperature=0.3  # Lower temperature for more precision
    )
)

# Helper function to process invoices
async def process_invoice(invoice_text):
    result = await Runner.run(agent, input=invoice_text)
    return result.final_output
'''

async def setup_integration_library():
    """Create and populate a fresh integration library with all necessary components"""
    # Create a fresh library
    library_path = "complete_openai_integration.json"
    
    # Delete existing file if it exists
    if os.path.exists(library_path):
        os.remove(library_path)
        print(f"Deleted existing library at {library_path} to create fresh version")
    
    # Initialize
    library = SmartLibrary(library_path)
    llm_service = LLMService(provider="openai", model="gpt-4o")
    
    print("Setting up components for OpenAI integration...")
    
    # 1. Add a DocumentAnalyzer tool
    document_analyzer_record = await library.create_record(
        name="DocumentAnalyzer",
        record_type="TOOL",
        domain="document_processing",
        description="Tool that analyzes and identifies the type of document from its content",
        code_snippet=DOCUMENT_ANALYZER_CODE,
        metadata={"framework": "beeai"},
        tags=["document", "analysis", "classification"]
    )
    print("✓ Created DocumentAnalyzer tool")
    
    # 2. Add a CalculationVerifier tool
    calculation_verifier_record = await library.create_record(
        name="CalculationVerifier",
        record_type="TOOL",
        domain="document_processing",
        description="Tool that verifies calculations in invoices, ensuring subtotal + tax = total",
        code_snippet=CALCULATION_VERIFIER_CODE,
        metadata={"framework": "beeai"},
        tags=["invoice", "calculation", "verification"]
    )
    print("✓ Created CalculationVerifier tool")
    
    # 3. Add an InvoiceSummaryGenerator tool
    invoice_summary_record = await library.create_record(
        name="InvoiceSummaryGenerator",
        record_type="TOOL",
        domain="document_processing",
        description="Tool that generates a concise summary of an invoice's key information",
        code_snippet=INVOICE_SUMMARY_GENERATOR_CODE,
        metadata={"framework": "beeai"},
        tags=["invoice", "summary", "report"]
    )
    print("✓ Created InvoiceSummaryGenerator tool")
    
    # 4. Add OpenAI invoice processor agent with OpenAI format
    openai_processor_record = await library.create_record(
        name="OpenAIInvoiceProcessor",
        record_type="AGENT",
        domain="document_processing",
        description=(
            "An OpenAI agent specialized in processing invoice documents. "
            "This agent can extract structured data from invoices, verify calculations, "
            "identify key information, and follow governance rules for financial documents."
        ),
        code_snippet=OPENAI_INVOICE_PROCESSOR_CODE,
        metadata={
            "framework": "openai-agents",
            "model": "gpt-4o",
            "model_settings": {
                "temperature": 0.3
            },
            "guardrails_enabled": True
        },
        tags=["openai", "invoice", "financial", "agent"]
    )
    print("✓ Created OpenAIInvoiceProcessor agent")
    
    print(f"\nLibrary setup complete at: {library_path}")
    return library_path

async def main():
    try:
        print("\n" + "="*80)
        print("COMPLETE OPENAI AGENTS INTEGRATION WITH SYSTEM AGENT")
        print("="*80)
        
        # Initialize components - first set up the library
        library_path = await setup_integration_library()
        
        # Initialize core components
        library = SmartLibrary(library_path)
        llm_service = LLMService(provider="openai", model="gpt-4o")
        agent_bus = SmartAgentBus(
            smart_library=library,
            system_agent=None,  # We'll set this after system_agent is created
            storage_path="smart_agent_bus.json", 
            log_path="agent_bus_logs.json"
        )
        
        # Set up provider registry with both providers
        provider_registry = ProviderRegistry()
        provider_registry.register_provider(BeeAIProvider(llm_service))
        provider_registry.register_provider(OpenAIAgentsProvider(llm_service))
        
        # Create factories
        tool_factory = ToolFactory(library, llm_service)
        agent_factory = AgentFactory(library, llm_service, provider_registry)
        
        # Create the SystemAgent
        print("\nInitializing SystemAgent...")
        system_agent = await SystemAgentFactory.create_agent(
            llm_service=llm_service,
            smart_library=library,
            agent_bus=agent_bus
        )

        # Update the agent_bus with the system_agent
        agent_bus.system_agent = system_agent

        # Initialize from library
        await agent_bus.initialize_from_library()
        
        # Initialize workflow components
        workflow_generator = WorkflowGenerator(llm_service, library, system_agent)
        workflow_processor = WorkflowProcessor(system_agent)
        
        # Attach workflow tools to the system agent
        system_agent.workflow_generator = workflow_generator
        system_agent.workflow_processor = workflow_processor
        
        print("SystemAgent initialized successfully!")
        
        # First, let's register our components with the Agent Bus to make them available
        print("\n" + "-"*80)
        print("PHASE 1: REGISTER COMPONENTS WITH AGENT BUS")
        print("-"*80)
        
        register_prompt = "Search for all available document processing components in the library, and register each one with the Agent Bus to make them accessible for our workflow. For each component: 1. Determine if it's a tool or an agent 2. Create an instance of it using the appropriate factory 3. Register it with the Agent Bus 4. Confirm registration was successful. List all registered components when done."
        
        print("\nPrompting SystemAgent to register components...")
        register_response = await system_agent.run(register_prompt)
        
        print("\nComponent Registration Result:")
        print(register_response.result.text)
        
        # Let's generate a workflow for invoice processing
        print("\n" + "-"*80)
        print("PHASE 2: GENERATE INVOICE PROCESSING WORKFLOW")
        print("-"*80)
        
        workflow_requirements = "Create a workflow for processing invoice documents with the following steps: 1. Use the DocumentAnalyzer tool to identify the type of document 2. If it's an invoice, use the OpenAIInvoiceProcessor agent to extract structured data 3. Use the CalculationVerifier tool to verify that subtotal + tax = total 4. Use the InvoiceSummaryGenerator tool to generate a concise summary. Here's a sample invoice to consider: " + SAMPLE_INVOICE[:200] + "... Make sure to use the exact component names from our library."
        
        print("Generating workflow YAML...")
        os.makedirs("workflows", exist_ok=True)
        workflow_yaml = await workflow_generator.generate_workflow(
            requirements=workflow_requirements,
            domain="document_processing",
            output_path="workflows/invoice_workflow.yaml"
        )
        
        print("\nGenerated Workflow YAML:")
        print(workflow_yaml)
        
        # Process an invoice with individual component execution
        print("\n" + "-"*80)
        print("PHASE 3: EXECUTE INDIVIDUAL COMPONENTS")
        print("-"*80)
        
        # First, analyze the document type
        analyze_prompt = "Execute the DocumentAnalyzer tool with the following invoice: " + SAMPLE_INVOICE + " The tool should identify the document type and extract key fields. Show the result."
        
        print("\nAnalyzing document type...")
        analyze_response = await system_agent.run(analyze_prompt)
        
        print("\nDocument Analysis Result:")
        print(analyze_response.result.text)
        
        # Process the invoice with the OpenAI agent
        process_prompt = "Now execute the OpenAIInvoiceProcessor agent to extract structured data from the following invoice: " + SAMPLE_INVOICE + " Return the structured data that includes invoice number, date, vendor, items, subtotal, tax, and total."
        
        print("\nProcessing invoice with OpenAI agent...")
        process_response = await system_agent.run(process_prompt)
        
        # Save the extracted data for next steps
        print("\nStructured Invoice Data:")
        print(process_response.result.text)
        
        # Execute the workflow
        print("\n" + "-"*80)
        print("PHASE 4: EXECUTE COMPLETE WORKFLOW")
        print("-"*80)
        
        print("\nProcessing the complete workflow...")
        workflow_result = await workflow_processor.process_workflow(workflow_yaml)
        
        print("\nComplete Workflow Execution Result:")
        print(workflow_result["result"])
        
        # Get final feedback
        print("\n" + "-"*80)
        print("PHASE 5: INTEGRATION ASSESSMENT")
        print("-"*80)
        
        feedback_prompt = "Based on the invoice processing we've done with both OpenAI and BeeAI components, please provide an assessment of: 1. The effectiveness of the integration between frameworks 2. The benefits of using OpenAI agents for this task 3. How the governance (firmware/guardrails) works across frameworks 4. Recommendations for further improving the integration. Focus specifically on how the agent-centric philosophy of our framework works with the OpenAI Agents SDK."
        
        print("\nRequesting integration assessment...")
        feedback_response = await system_agent.run(feedback_prompt)
        
        print("\nIntegration Assessment:")
        print(feedback_response.result.text)
        
        print("\nDemonstration completed successfully!")
        
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())