# examples/setup_simplified_agent_library.py

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

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Define real BeeAI-compatible tool and agent code snippets

# LLM-based DocumentAnalyzer tool
DOCUMENT_ANALYZER_TOOL = '''
from typing import Dict, Any
import json
from pydantic import BaseModel, Field

from beeai_framework.context import RunContext
from beeai_framework.emitter.emitter import Emitter
from beeai_framework.tools.tool import StringToolOutput, Tool, ToolRunOptions
from beeai_framework.backend.chat import ChatModel
from beeai_framework.backend.message import UserMessage

class DocumentAnalyzerInput(BaseModel):
    text: str = Field(description="Document text to analyze")

class DocumentAnalyzer(Tool[DocumentAnalyzerInput, ToolRunOptions, StringToolOutput]):
    """
    Analyzes a document to identify its type and key characteristics using an LLM.
    """
    name = "DocumentAnalyzer"
    description = "Identifies document type and extracts key information"
    input_schema = DocumentAnalyzerInput

    def __init__(self, options: Dict[str, Any] | None = None):
        super().__init__(options=options or {})
        # Get a chat model from the options or create a default one
        self.chat_model = options.get("chat_model") if options else None
        
    def _create_emitter(self) -> Emitter:
        return Emitter.root().child(
            namespace=["tool", "document", "analyzer"],
            creator=self,
        )
    
    async def _run(self, input: DocumentAnalyzerInput, options: ToolRunOptions | None, context: RunContext) -> StringToolOutput:
        """
        Analyzes a document using an LLM to identify its type and extract key information.
        
        Args:
            input: Document text to analyze
            
        Returns:
            Document analysis including type, confidence, and keywords
        """
        # If we don't have a chat model, try to get one from the context
        if not self.chat_model and context and hasattr(context, "llm"):
            self.chat_model = context.llm
        
        # If we still don't have a chat model, try to get the default one
        if not self.chat_model:
            try:
                from beeai_framework.backend.chat import get_default_chat_model
                self.chat_model = get_default_chat_model()
            except:
                pass
                
        # Fall back to OpenAI if available
        if not self.chat_model:
            try:
                from beeai_framework.adapters.litellm.chat import LiteLLMChatModel
                self.chat_model = LiteLLMChatModel("gpt-4o", provider_id="openai")
            except:
                return StringToolOutput(json.dumps({
                    "error": "No chat model available for document analysis"
                }))
        
        # Create the prompt for document analysis
        prompt = f"""
        Please analyze the following document and provide structured information about it.
        Identify the document type, extract key information, and provide a confidence score.
        
        Return the results in JSON format with the following structure:
        {{
            "document_type": "Type of document (invoice, medical_record, contract, etc.)",
            "confidence": 0.0-1.0,
            "keywords": ["list", "of", "key", "words"],
            "extracted_data": {{
                "field1": "value1",
                "field2": "value2",
                ...
            }}
        }}
        
        DOCUMENT TO ANALYZE:
        {input.text}
        """
        
        try:
            # Query the LLM
            message = UserMessage(prompt)
            response = await self.chat_model.create(messages=[message])
            response_text = response.get_text_content()
            
            # Try to parse the response as JSON
            try:
                result = json.loads(response_text)
                # Ensure we have all the required fields
                if not isinstance(result, dict):
                    result = {"document_type": "unknown", "error": "Invalid response format"}
                if "document_type" not in result:
                    result["document_type"] = "unknown"
                if "confidence" not in result:
                    result["confidence"] = 0.5
                if "keywords" not in result:
                    result["keywords"] = []
                if "extracted_data" not in result:
                    result["extracted_data"] = {}
            except json.JSONDecodeError:
                # If the response isn't valid JSON, try to extract JSON from it
                import re
                json_match = re.search(r'dummy', response_text, re.DOTALL)
                if json_match:
                    try:
                        result = json.loads(json_match.group(0))
                    except:
                        result = {
                            "document_type": "unknown",
                            "confidence": 0.5,
                            "keywords": [],
                            "extracted_data": {},
                            "raw_response": response_text[:500]  # Include part of the raw response
                        }
                else:
                    # Create a basic response
                    result = {
                        "document_type": "unknown",
                        "confidence": 0.5,
                        "keywords": [],
                        "raw_response": response_text[:500]  # Include part of the raw response
                    }
            
            return StringToolOutput(json.dumps(result, indent=2))
            
        except Exception as e:
            error_result = {
                "error": f"Error analyzing document: {str(e)}",
                "document_type": "unknown",
                "confidence": 0.0
            }
            return StringToolOutput(json.dumps(error_result, indent=2))
'''

# LLM-based AgentCommunicator tool
AGENT_COMMUNICATOR_TOOL = '''
from typing import Dict, Any, Optional
import json
from pydantic import BaseModel, Field

from beeai_framework.context import RunContext
from beeai_framework.emitter.emitter import Emitter
from beeai_framework.tools.tool import StringToolOutput, Tool, ToolRunOptions
from beeai_framework.backend.chat import ChatModel
from beeai_framework.backend.message import UserMessage

class AgentCommunicatorInput(BaseModel):
    agent_name: str = Field(description="Name of the agent to communicate with")
    message: str = Field(description="The message/request to send")
    data: Dict[str, Any] = Field(description="Any supporting data to include", default_factory=dict)

class AgentCommunicator(Tool[AgentCommunicatorInput, ToolRunOptions, StringToolOutput]):
    """
    Facilitates communication between agents by formatting requests and routing them to specialized agents.
    """
    name = "AgentCommunicator"
    description = "Enables communication between different specialized agents"
    input_schema = AgentCommunicatorInput

    def __init__(self, options: Dict[str, Any] | None = None):
        super().__init__(options=options or {})
        # Get a chat model from the options or create a default one
        self.chat_model = options.get("chat_model") if options else None
        
    def _create_emitter(self) -> Emitter:
        return Emitter.root().child(
            namespace=["tool", "agent", "communicator"],
            creator=self,
        )
    
    async def _run(self, input: AgentCommunicatorInput, options: ToolRunOptions | None, context: RunContext) -> StringToolOutput:
        """
        Process a communication request between agents using an LLM.
        
        Args:
            input: Communication request details including agent name, message, and data
            
        Returns:
            Response from the requested agent
        """
        try:
            # Log the communication attempt
            agent_name = input.agent_name
            message = input.message
            data = input.data
            
            # If we don't have a chat model, try to get one from the context
            if not self.chat_model and context and hasattr(context, "llm"):
                self.chat_model = context.llm
            
            # If we still don't have a chat model, try to get the default one
            if not self.chat_model:
                try:
                    from beeai_framework.backend.chat import get_default_chat_model
                    self.chat_model = get_default_chat_model()
                except:
                    pass
                    
            # Fall back to OpenAI if available
            if not self.chat_model:
                try:
                    from beeai_framework.adapters.litellm.chat import LiteLLMChatModel
                    self.chat_model = LiteLLMChatModel("gpt-4o", provider_id="openai")
                except:
                    return StringToolOutput(json.dumps({
                        "error": "No chat model available for agent communication"
                    }))
            
            # Create specialized prompts based on which agent is being contacted
            if agent_name == "SpecialistAgent":
                # Create a prompt for the specialist agent
                prompt = f"""
                You are a specialist agent that performs detailed document analysis.
                
                Analyze the following document and provide a structured response.
                Return a JSON object with 'analysis' and 'extracted_data' fields.
                
                DOCUMENT TYPE: {data.get('document_type', 'unknown')}
                
                DOCUMENT CONTENT:
                {message}
                
                ADDITIONAL CONTEXT:
                {json.dumps(data, indent=2)}
                """
            else:
                # Generic communication
                prompt = f"""
                You are simulating agent '{agent_name}'.
                
                Please respond to the following message as if you were the agent:
                {message}
                
                ADDITIONAL CONTEXT:
                {json.dumps(data, indent=2)}
                
                Return your response in JSON format with appropriate fields.
                """
            
            # Query the LLM
            message_obj = UserMessage(prompt)
            response = await self.chat_model.create(messages=[message_obj])
            response_text = response.get_text_content()
            
            # Try to parse the response as JSON
            try:
                result = json.loads(response_text)
            except json.JSONDecodeError:
                # If the response isn't valid JSON, try to extract JSON from it
                import re
                json_match = re.search(r'dummy', response_text, re.DOTALL)
                if json_match:
                    try:
                        result = json.loads(json_match.group(0))
                    except:
                        # Structure the response manually
                        result = {
                            "analysis": {
                                "document_type": data.get("document_type", "unknown"),
                                "notes": "Structured response could not be extracted"
                            },
                            "extracted_data": {},
                            "raw_response": response_text[:500]  # Include part of the raw response
                        }
                else:
                    # Structure the response manually
                    result = {
                        "analysis": {
                            "document_type": data.get("document_type", "unknown"),
                            "notes": "Structured response could not be extracted"
                        },
                        "extracted_data": {},
                        "raw_response": response_text[:500]  # Include part of the raw response
                    }
            
            return StringToolOutput(json.dumps(result, indent=2))
            
        except Exception as e:
            error_result = {
                "error": f"Communication error: {str(e)}",
                "analysis": {
                    "document_type": data.get("document_type", "unknown") if isinstance(data, dict) else "unknown",
                    "success": False
                },
                "extracted_data": {}
            }
            return StringToolOutput(json.dumps(error_result, indent=2))
'''

# Real BeeAI-compatible SpecialistAgent
SPECIALIST_AGENT = '''
from typing import List, Dict, Any, Optional
import json
import re

from beeai_framework.agents.react import ReActAgent
from beeai_framework.agents.types import AgentMeta
from beeai_framework.memory import TokenMemory, UnconstrainedMemory
from beeai_framework.backend.chat import ChatModel
from beeai_framework.tools.tool import Tool

class SpecialistAgentInitializer:
    """
    Specialist agent that performs detailed document analysis.
    
    This agent provides deep expertise for specific document types,
    extracting important information and providing domain-specific insights.
    """
    
    @staticmethod
    def create_agent(llm: ChatModel, tools: Optional[List[Tool]] = None) -> ReActAgent:
        """Create and configure the specialist agent with tools."""
        # Use empty tools list if none provided
        if tools is None:
            tools = []
            
        # Define agent metadata
        meta = AgentMeta(
            name="SpecialistAgent",
            description=(
                "Specialist agent that performs detailed document analysis. "
                "This agent provides deep expertise for specific document types, "
                "extracting important information and providing domain-specific insights."
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
    async def analyze_document(document_text: str, document_type: Optional[str] = None) -> Dict[str, Any]:
        """
        Analyzes a document and returns structured information.
        
        Args:
            document_text: The text of the document to analyze
            document_type: Optional type hint for the document
            
        Returns:
            Structured analysis of the document
        """
        # Determine document type if not provided
        if not document_type:
            lower_text = document_text.lower()
            if "invoice" in lower_text:
                document_type = "invoice"
            elif "patient" in lower_text or "medical" in lower_text:
                document_type = "medical"
            else:
                document_type = "unknown"
        
        # Initialize results
        results = {
            "analysis": {
                "document_type": document_type,
                "priority": "medium"
            },
            "extracted_data": {}
        }
        
        # Extract common data
        # Dates (YYYY-MM-DD format)
        date_pattern = r'(\d{4}-\d{2}-\d{2})'
        dates = re.findall(date_pattern, document_text)
        if dates:
            results["extracted_data"]["dates"] = dates
        
        # Perform document-specific analysis
        if document_type == "invoice":
            # Analyze invoice
            SpecialistAgentInitializer._analyze_invoice(document_text, results)
        elif document_type == "medical":
            # Analyze medical record
            SpecialistAgentInitializer._analyze_medical_record(document_text, results)
        else:
            # Generic analysis for unknown types
            results["analysis"]["notes"] = "Document type not recognized for specialized analysis"
        
        return results
    
    @staticmethod
    def _analyze_invoice(text: str, results: Dict[str, Any]) -> None:
        """Specialized invoice analysis"""
        # Extract vendor
        vendor_match = re.search(r'Vendor: ([^\\n]+)', text)
        if vendor_match:
            results["extracted_data"]["vendor"] = vendor_match.group(1).strip()
        
        # Extract invoice number
        invoice_num_match = re.search(r'(?:INVOICE|Invoice)[ #:]+([A-Z0-9]+)', text)
        if invoice_num_match:
            results["extracted_data"]["invoice_number"] = invoice_num_match.group(1).strip()
        
        # Extract total amount
        total_match = re.search(r'Total[^:]*: ?\\$(\d+,?\d*\.\d{2})', text)
        if total_match:
            total = total_match.group(1).replace(",", "")
            results["extracted_data"]["total_amount"] = total
            
            # Set priority based on amount
            try:
                amount = float(total)
                if amount > 1000:
                    results["analysis"]["priority"] = "high"
                    results["analysis"]["approval_required"] = True
                    results["analysis"]["notes"] = "Large invoice requires manager approval"
                else:
                    results["analysis"]["approval_required"] = False
            except:
                pass
        
        # Extract due date
        due_date_match = re.search(r'Due Date: (\d{4}-\d{2}-\d{2})', text)
        if due_date_match:
            results["extracted_data"]["due_date"] = due_date_match.group(1)
    
    @staticmethod
    def _analyze_medical_record(text: str, results: Dict[str, Any]) -> None:
        """Specialized medical record analysis"""
        # Extract patient name
        name_match = re.search(r'Name: ([^\\n]+)', text)
        if name_match:
            results["extracted_data"]["patient_name"] = name_match.group(1).strip()
        
        # Extract patient ID
        patient_id_match = re.search(r'Patient ID: ([^\\n]+)', text)
        if patient_id_match:
            results["extracted_data"]["patient_id"] = patient_id_match.group(1).strip()
        
        # Extract diagnosis
        diagnosis_match = re.search(r'Assessment: ([^\\n]+)', text)
        if diagnosis_match:
            diagnosis = diagnosis_match.group(1).strip()
            results["extracted_data"]["diagnosis"] = diagnosis
            
            # Set priority based on diagnosis severity
            if "acute" in diagnosis.lower() or "emergency" in diagnosis.lower():
                results["analysis"]["priority"] = "high"
                results["analysis"]["follow_up_required"] = True
                results["analysis"]["notes"] = "Urgent condition requires immediate follow-up"
            else:
                results["analysis"]["follow_up_required"] = False
        
        # Extract vital signs
        vitals = {}
        temp_match = re.search(r'Temperature: ([^\\n]+)', text)
        if temp_match:
            temp = temp_match.group(1).strip()
            vitals["temperature"] = temp
            
            # Flag fever
            if "°F" in temp and float(temp.replace("°F", "").strip()) > 100:
                results["analysis"]["has_fever"] = True
        
        bp_match = re.search(r'Blood Pressure: ([^\\n]+)', text)
        if bp_match:
            vitals["blood_pressure"] = bp_match.group(1).strip()
        
        if vitals:
            results["extracted_data"]["vitals"] = vitals
'''

# Real BeeAI-compatible CoordinatorAgent
COORDINATOR_AGENT = '''
from typing import List, Dict, Any, Optional
import json

from beeai_framework.agents.react import ReActAgent
from beeai_framework.agents.types import AgentMeta
from beeai_framework.memory import TokenMemory, UnconstrainedMemory
from beeai_framework.backend.chat import ChatModel
from beeai_framework.tools.tool import Tool

class CoordinatorAgentInitializer:
    """
    Primary agent that orchestrates document processing.
    
    This agent:
    1. Analyzes the document to identify its type
    2. Sends the document to a specialist agent for detailed analysis
    3. Synthesizes the results and provides recommendations
    """
    
    @staticmethod
    def create_agent(llm: ChatModel, tools: Optional[List[Tool]] = None) -> ReActAgent:
        """Create and configure the coordinator agent with tools."""
        # Use empty tools list if none provided
        if tools is None:
            tools = []
            
        # Define agent metadata
        meta = AgentMeta(
            name="CoordinatorAgent",
            description=(
                "Primary agent that orchestrates document processing. "
                "This agent analyzes documents, delegates to specialists, "
                "and synthesizes results with recommendations."
            ),
            tools=tools
        )
        
        # Create the agent with the necessary tools
        agent = ReActAgent(
            llm=llm,
            tools=tools,
            memory=TokenMemory(llm),
            meta=meta
        )
        
        return agent
        
    @staticmethod
    def generate_recommendations(document_type: str, analysis: Dict[str, Any], extracted_data: Dict[str, Any]) -> List[str]:
        """
        Generate recommendations based on document analysis.
        
        Args:
            document_type: Type of document (invoice, medical, etc.)
            analysis: Analysis data from specialist agent
            extracted_data: Data extracted from the document
            
        Returns:
            List of recommendations
        """
        recommendations = []
        
        if document_type == "invoice":
            # Invoice recommendations
            if "total_amount" in extracted_data:
                try:
                    amount = float(extracted_data["total_amount"])
                    if amount > 1000:
                        recommendations.append("HIGH PRIORITY: Review large invoice amount")
                    
                    if "approval_required" in analysis and analysis["approval_required"]:
                        recommendations.append("Route to finance manager for approval")
                    else:
                        recommendations.append("Process for payment within standard timeframe")
                except:
                    recommendations.append("Verify invoice amount")
            
            if "vendor" in extracted_data:
                recommendations.append(f"Confirm vendor details for {extracted_data['vendor']}")
                
            if "dates" in extracted_data and len(extracted_data["dates"]) > 0:
                recommendations.append(f"Document date: {extracted_data['dates'][0]}")
        
        elif document_type == "medical":
            # Medical record recommendations
            if "diagnosis" in extracted_data:
                recommendations.append(f"Noted diagnosis: {extracted_data['diagnosis']}")
                
            if "follow_up_required" in analysis and analysis["follow_up_required"]:
                recommendations.append("PRIORITY: Schedule follow-up appointment")
            
            if "patient_name" in extracted_data:
                recommendations.append(f"Update patient record for {extracted_data['patient_name']}")
        
        else:
            # Default recommendations
            recommendations.append("Document requires manual review")
            recommendations.append("Route to appropriate department based on content")
        
        # Add standard recommendation
        recommendations.append("Archive document according to record retention policy")
        
        return recommendations
'''

async def main():
    try:
        # Check if library exists, and if so, delete it to recreate
        library_path = "simplified_agent_library.json"
        if os.path.exists(library_path):
            os.remove(library_path)
            print(f"Deleted existing library at {library_path} to create a fresh version")
            
        # Initialize library
        library = SmartLibrary(library_path)
        llm_service = LLMService(provider="openai", model="gpt-4o")
        
        print("Setting up simplified agent library with real BeeAI agents and tools...")
        
        # Add the AgentCommunicator tool
        await library.create_record(
            name="AgentCommunicator",
            record_type="TOOL",
            domain="document_processing",
            description="Tool for facilitating communication between agents",
            code_snippet=AGENT_COMMUNICATOR_TOOL,
            tags=["communication", "agent", "tool"]
        )
        print("✓ Created AgentCommunicator tool (BeeAI-compatible)")
        
        # Add the DocumentAnalyzer tool
        await library.create_record(
            name="DocumentAnalyzer",
            record_type="TOOL",
            domain="document_processing",
            description="Tool to analyze documents and identify their type",
            code_snippet=DOCUMENT_ANALYZER_TOOL,
            tags=["analysis", "tool"]
        )
        print("✓ Created DocumentAnalyzer tool (BeeAI-compatible)")
        
        # Add the SpecialistAgent 
        await library.create_record(
            name="SpecialistAgent",
            record_type="AGENT",
            domain="document_processing",
            description="Specialist agent that performs detailed document analysis",
            code_snippet=SPECIALIST_AGENT,
            metadata={"framework": "beeai"},
            tags=["specialist", "agent", "analysis"]
        )
        print("✓ Created SpecialistAgent (BeeAI-compatible)")
        
        # Add the CoordinatorAgent 
        await library.create_record(
            name="CoordinatorAgent",
            record_type="AGENT",
            domain="document_processing",
            description="Primary agent that orchestrates document processing",
            code_snippet=COORDINATOR_AGENT,
            metadata={
                "framework": "beeai",
                "required_tools": ["DocumentAnalyzer", "AgentCommunicator"]
            },
            tags=["coordinator", "agent", "orchestration"]
        )
        print("✓ Created CoordinatorAgent (BeeAI-compatible)")
        
        print("\nLibrary setup complete with real BeeAI agents and tools!")
        print(f"Library saved to: {library.storage_path}")
        
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())