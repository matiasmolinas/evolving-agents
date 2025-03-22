# generate_smart_library.py
import json
import os
import asyncio
from typing import Dict, Any, List, Optional

from evolving_agents.core.llm_service import LLMService
from evolving_agents.smart_library.smart_library import SmartLibrary

async def generate_financial_library() -> Dict[str, Any]:
    """Generate the smart library for financial examples."""
    # Create a temporary library
    llm_service = LLMService(provider="openai", model="gpt-4o")
    smart_library = SmartLibrary("temp_financial_library.json", llm_service)
    
    # Create document analyzer component
    document_analyzer = {
        "name": "DocumentAnalyzer",
        "record_type": "TOOL",
        "domain": "document_processing",
        "description": "Tool that analyzes and identifies the type of document from its content",
        "code_snippet": """
# Document analyzer code here
""",
        "version": "1.0.0",
        "tags": ["document", "analysis", "classification"],
        "capabilities": [
            {
                "id": "document_analysis",
                "name": "Document Analysis",
                "description": "Analyzes and identifies the type of document from its content",
                "context": {
                    "required_fields": ["document_text"],
                    "produced_fields": ["document_type", "confidence", "extracted_fields"]
                }
            }
        ],
        "metadata": {"framework": "beeai"}
    }
    
    # Create calculation verifier component
    calculation_verifier = {
        "name": "CalculationVerifier",
        "record_type": "TOOL",
        "domain": "document_processing",
        "description": "Tool that verifies calculations in invoices, ensuring subtotal + tax = total",
        "code_snippet": """
# Calculation verifier code here
""",
        "version": "1.0.0",
        "tags": ["invoice", "calculation", "verification"],
        "capabilities": [
            {
                "id": "calculation_verification",
                "name": "Calculation Verification",
                "description": "Verifies that calculations in an invoice are correct (subtotal + tax = total)",
                "context": {
                    "required_fields": ["invoice_data"],
                    "produced_fields": ["is_correct", "expected_total", "difference"]
                }
            }
        ],
        "metadata": {"framework": "beeai"}
    }
    
    # Add more financial components...
    invoice_summary = {
        "name": "InvoiceSummaryGenerator",
        "record_type": "TOOL",
        "domain": "document_processing",
        "description": "Tool that generates a concise summary of an invoice's key information",
        "code_snippet": """
# Invoice summary generator code here
""",
        "version": "1.0.0",
        "tags": ["invoice", "summary", "report"],
        "capabilities": [
            {
                "id": "summary_generation",
                "name": "Summary Generation",
                "description": "Generates a concise summary of an invoice with key details and recommendations",
                "context": {
                    "required_fields": ["invoice_data"],
                    "produced_fields": ["summary", "key_details", "recommendations"]
                }
            }
        ],
        "metadata": {"framework": "beeai"}
    }
    
    openai_invoice_processor = {
        "name": "OpenAIInvoiceProcessor",
        "record_type": "AGENT",
        "domain": "document_processing",
        "description": "An OpenAI agent specialized in processing invoice documents",
        "code_snippet": """
# OpenAI invoice processor code here
""",
        "version": "1.0.0",
        "tags": ["openai", "invoice", "financial", "agent"],
        "capabilities": [
            {
                "id": "invoice_data_extraction",
                "name": "Invoice Data Extraction",
                "description": "Extracts structured data from invoice documents",
                "context": {
                    "required_fields": ["invoice_text"],
                    "produced_fields": ["structured_invoice_data"]
                }
            }
        ],
        "metadata": {
            "framework": "openai-agents",
            "model": "gpt-4o",
            "model_settings": {"temperature": 0.3},
            "guardrails_enabled": True
        }
    }
    
    # Add the components to the library
    await smart_library.create_record(**document_analyzer)
    await smart_library.create_record(**calculation_verifier)
    await smart_library.create_record(**invoice_summary)
    await smart_library.create_record(**openai_invoice_processor)
    
    # Create the library json structure
    library_json = {
        "records": smart_library.records
    }
    
    # Clean up temp file
    if os.path.exists("temp_financial_library.json"):
        os.remove("temp_financial_library.json")
    
    return library_json

async def generate_medical_library() -> Dict[str, Any]:
    """Generate the smart library for medical examples."""
    # Create a temporary library
    llm_service = LLMService(provider="openai", model="gpt-4o")
    smart_library = SmartLibrary("temp_medical_library.json", llm_service)
    
    # BMI Calculator
    bmi_calculator = {
        "name": "BMICalculator",
        "record_type": "TOOL",
        "domain": "medical_assessment",
        "description": "Tool that calculates Body Mass Index (BMI) and classifies weight status",
        "code_snippet": """
# BMI Calculator code here
""",
        "version": "1.0.0",
        "tags": ["medical", "assessment", "bmi", "calculator"],
        "capabilities": [
            {
                "id": "bmi_calculation",
                "name": "BMI Calculation",
                "description": "Calculates Body Mass Index from height and weight and classifies weight status",
                "context": {
                    "required_fields": ["weight_kg", "height_cm"],
                    "produced_fields": ["bmi", "weight_status", "health_risk"]
                }
            }
        ],
        "metadata": {"framework": "beeai"}
    }
    
    # Cardiovascular Risk Calculator
    cv_risk_calculator = {
        "name": "CardiovascularRiskCalculator",
        "record_type": "TOOL",
        "domain": "medical_assessment",
        "description": "Tool that calculates 10-year cardiovascular disease risk using established medical formulas",
        "code_snippet": """
# Cardiovascular risk calculator code here
""",
        "version": "1.0.0",
        "tags": ["medical", "cardiology", "risk_assessment", "framingham"],
        "capabilities": [
            {
                "id": "cardiovascular_risk_calculation",
                "name": "Cardiovascular Risk Calculation",
                "description": "Calculates 10-year cardiovascular disease risk using established medical formulas",
                "context": {
                    "required_fields": ["age", "sex", "total_cholesterol", "hdl_cholesterol", "systolic_bp", "is_bp_treated", "is_smoker", "has_diabetes"],
                    "produced_fields": ["risk_percentage", "risk_level", "recommendations"]
                }
            }
        ],
        "metadata": {"framework": "beeai"}
    }
    
    # Physiological Data Extractor
    phys_data_extractor = {
        "name": "PhysiologicalDataExtractor",
        "record_type": "AGENT",
        "domain": "medical_assessment",
        "description": "An agent that extracts structured physiological data from medical records for analysis",
        "code_snippet": """
# Physiological data extractor code here
""",
        "version": "1.0.0",
        "tags": ["medical", "extraction", "physiological", "data"],
        "capabilities": [
            {
                "id": "physiological_data_extraction",
                "name": "Physiological Data Extraction",
                "description": "Extracts structured physiological data from medical records",
                "context": {
                    "required_fields": ["medical_record_text"],
                    "produced_fields": ["structured_physiological_data", "vital_signs", "lab_values", "cardiovascular_risk_factors"]
                }
            }
        ],
        "metadata": {"framework": "beeai"}
    }
    
    # Medical Analysis Agent
    medical_analyzer = {
        "name": "MedicalAnalysisAgent",
        "record_type": "AGENT",
        "domain": "medical_assessment",
        "description": "An agent that analyzes physiological data and risk scores to provide clinical interpretations and recommendations",
        "code_snippet": """
# Medical analysis agent code here
""",
        "version": "1.0.0",
        "tags": ["medical", "analysis", "interpretation", "assessment"],
        "capabilities": [
            {
                "id": "medical_data_analysis",
                "name": "Medical Data Analysis",
                "description": "Analyzes physiological data and risk scores to provide clinical interpretations and recommendations",
                "context": {
                    "required_fields": ["structured_physiological_data", "risk_scores"],
                    "produced_fields": ["clinical_analysis", "interpretations", "recommendations", "flags"]
                }
            }
        ],
        "metadata": {"framework": "beeai"}
    }
    
    # Add the components to the library
    await smart_library.create_record(**bmi_calculator)
    await smart_library.create_record(**cv_risk_calculator)
    await smart_library.create_record(**phys_data_extractor)
    await smart_library.create_record(**medical_analyzer)
    
    # Create the library json structure
    library_json = {
        "records": smart_library.records
    }
    
    # Clean up temp file
    if os.path.exists("temp_medical_library.json"):
        os.remove("temp_medical_library.json")
    
    return library_json

async def generate_libraries():
    """Generate and save both libraries."""
    # Generate libraries
    financial_library = await generate_financial_library()
    medical_library = await generate_medical_library()
    
    # Save to files
    os.makedirs("data", exist_ok=True)
    
    with open("data/financial_library.json", "w") as f:
        json.dump(financial_library, f, indent=2)
    
    with open("data/medical_library.json", "w") as f:
        json.dump(medical_library, f, indent=2)
    
    print("Libraries generated and saved to data/financial_library.json and data/medical_library.json")

if __name__ == "__main__":
    asyncio.run(generate_libraries())