# examples/self_improvement/evolve_smart_library.py

import asyncio
import logging
import os
import json
import re
from typing import Dict, Any, List, Optional, Union

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import toolkit components
from evolving_agents.core.dependency_container import DependencyContainer
from evolving_agents.core.llm_service import LLMService
from evolving_agents.smart_library.smart_library import SmartLibrary
from evolving_agents.agent_bus.smart_agent_bus import SmartAgentBus
from evolving_agents.core.system_agent import SystemAgentFactory
from evolving_agents.firmware.firmware import Firmware

def extract_code_from_response(response_text: str) -> Optional[str]:
    """Extract code from LLM response using multiple strategies."""
    # Strategy 1: Look for ```python...``` code blocks
    code_pattern = r"```python\s*(.*?)\s*```"
    matches = re.findall(code_pattern, response_text, re.DOTALL)
    
    if matches:
        return matches[0].strip()
    
    # Strategy 2: Look for any code blocks
    generic_code_pattern = r"```\s*(.*?)\s*```"
    matches = re.findall(generic_code_pattern, response_text, re.DOTALL)
    
    if matches:
        # Check if it looks like Python code
        for match in matches:
            if 'class' in match and 'def' in match:
                return match.strip()
    
    # Strategy 3: Look for class definition
    class_pattern = r"(class\s+EnhancedSmartLibrary.*?(?:def\s+__init__|\Z))"
    matches = re.findall(class_pattern, response_text, re.DOTALL)
    
    if matches:
        return matches[0].strip()
    
    logger.warning("Could not extract code using standard patterns")
    
    # Strategy 4: Try to extract any Python-like content as a last resort
    if "class EnhancedSmartLibrary" in response_text:
        start_idx = response_text.find("class EnhancedSmartLibrary")
        # Try to find a reasonable end point
        end_markers = ["# End of EnhancedSmartLibrary", "# Testing the EnhancedSmartLibrary"]
        end_idx = len(response_text)
        
        for marker in end_markers:
            marker_idx = response_text.find(marker, start_idx)
            if marker_idx != -1 and marker_idx < end_idx:
                end_idx = marker_idx
        
        return response_text[start_idx:end_idx].strip()
    
    return None

async def evolve_smart_library():
    """Demonstrate self-improvement by enhancing the SmartLibrary component."""
    
    # Setup clean test environment
    library_path = "self_improvement_library.json"
    if os.path.exists(library_path):
        os.remove(library_path)
    
    # Create dependency container
    container = DependencyContainer()
    
    # Initialize core components
    llm_service = LLMService(provider="openai", model="gpt-4o")
    container.register('llm_service', llm_service)
    
    # Create library with initial state
    smart_library = SmartLibrary(library_path, llm_service=llm_service, container=container)
    container.register('smart_library', smart_library)
    
    # Create firmware component
    firmware = Firmware()
    container.register('firmware', firmware)
    
    # Create agent bus
    agent_bus = SmartAgentBus(
        storage_path="self_improvement_bus.json",
        log_path="self_improvement_logs.json",
        container=container
    )
    container.register('agent_bus', agent_bus)
    
    # Create system agent
    system_agent = await SystemAgentFactory.create_agent(container=container)
    container.register('system_agent', system_agent)
    
    # Initialize components
    await smart_library.initialize()
    await agent_bus.initialize_from_library()
    
    # First, let's analyze the current implementation of semantic search in SmartLibrary
    logger.info("PHASE 1: Analyzing current SmartLibrary semantic search implementation")
    
    analysis_prompt = """
    I need you to analyze the semantic search capabilities of the SmartLibrary component. 
    
    The SmartLibrary has a method called `semantic_search` that finds components semantically similar to a query.
    
    Current implementation details:
    1. Uses vector embeddings from an LLM service
    2. Supports basic filtering by record_type and domain
    3. Has minor boosting based on usage metrics and success rate
    4. Returns (record, similarity) tuples sorted by similarity
    
    Please analyze limitations of this approach and identify specific enhancement opportunities.
    Provide concrete technical suggestions that would make semantic search more powerful and accurate.
    """
    
    analysis_result = await system_agent.run(analysis_prompt)
    logger.info("Analysis complete. Identified enhancement opportunities.")
    
    # Now let's be very explicit about what we want the System Agent to evolve
    logger.info("\nPHASE 2: Creating improved SmartLibrary with enhanced semantic search")
    
    evolution_prompt = """
    I need you to create an enhanced version of the SmartLibrary component to improve its semantic search capabilities.
    
    Create a new `EnhancedSmartLibrary` class that inherits from SmartLibrary and overrides the `semantic_search` method.
    
    Your implementation must include these specific improvements:
    
    1. Contextual domain boosting: Increase similarity scores for records that match the query's domain
    2. Recency weighting: Add small boosts to more recently created or updated components
    3. Tag-based boosting: Increase scores when query terms match component tags
    4. Multi-field search: Consider component name, description, and tags when computing relevance
    5. Explanation generation: For each result, provide a brief explanation of why it matched
    
    Your code must be fully functional, properly documented, and follow these requirements:
    - Include imports needed from SmartLibrary
    - Override `semantic_search` while maintaining its signature
    - Call the parent class method as part of implementation
    - Properly handle edge cases like empty results
    
    Return ONLY the Python code for the EnhancedSmartLibrary class WITHOUT additional explanation.
    The code must be executable and compatible with the existing toolkit architecture.
    """
    
    evolution_result = await system_agent.run(evolution_prompt)
    logger.info("Evolution complete. Enhanced SmartLibrary implementation received.")
    
    # Extract the evolved SmartLibrary code using our improved extractor
    logger.info("\nPHASE 3: Extracting and saving the evolved component code")
    
    evolved_code = extract_code_from_response(evolution_result.result.text)
    
    if evolved_code:
        # Save the evolved code
        enhanced_library_path = "enhanced_smart_library.py"
        with open(enhanced_library_path, "w") as f:
            f.write(evolved_code)
        logger.info(f"Enhanced SmartLibrary code saved to {enhanced_library_path}")
        
        # Also save it as a component in the library
        await smart_library.create_record(
            name="EnhancedSmartLibrary",
            record_type="TOOL",
            domain="core_framework",
            description="Enhanced version of SmartLibrary with improved semantic search capabilities",
            code_snippet=evolved_code,
            tags=["enhanced", "semantic_search", "library"]
        )
        logger.info("EnhancedSmartLibrary component saved to SmartLibrary")
    else:
        logger.error("Failed to extract code from evolution result.")
        with open("raw_evolution_response.txt", "w") as f:
            f.write(evolution_result.result.text)
        logger.info("Raw response saved to raw_evolution_response.txt for inspection")
        return
    
    # Now let's seed the library with test components
    logger.info("\nPHASE 4: Preparing test data for search comparison")
    
    # Seed the SmartLibrary with diverse test components
    await smart_library.create_record(
        name="InvoiceProcessor",
        record_type="AGENT",
        domain="finance",
        description="Agent for processing and extracting data from invoice documents",
        code_snippet="# Invoice processing code",
        tags=["invoice", "finance", "document", "extraction"]
    )
    
    await smart_library.create_record(
        name="ContractAnalyzer",
        record_type="AGENT",
        domain="legal",
        description="Agent for analyzing legal contracts and extracting key terms",
        code_snippet="# Contract analysis code",
        tags=["contract", "legal", "document", "analysis"]
    )
    
    await smart_library.create_record(
        name="DocumentClassifier",
        record_type="TOOL",
        domain="document_processing",
        description="Tool for classifying document types based on content analysis",
        code_snippet="# Document classification code",
        tags=["document", "classification", "ai"]
    )
    
    await smart_library.create_record(
        name="CustomerFeedbackAnalyzer",
        record_type="AGENT",
        domain="customer_service",
        description="Agent that analyzes customer feedback and identifies sentiment and key issues",
        code_snippet="# Sentiment analysis code",
        tags=["feedback", "sentiment", "analysis"]
    )
    
    await smart_library.create_record(
        name="FinancialReportGenerator",
        record_type="TOOL",
        domain="finance",
        description="Tool for generating financial reports from raw data",
        code_snippet="# Financial report generation code",
        tags=["finance", "reports", "visualization"]
    )
    
    # Now execute a comparison test using System Agent
    logger.info("\nPHASE 5: Testing and comparing the enhanced semantic search")
    
    comparison_prompt = f"""
    I need you to test and evaluate the enhanced semantic search capability in EnhancedSmartLibrary.
    
    First, here is the code for the enhanced implementation:
    
    ```python
    {evolved_code}
    ```
    
    The SmartLibrary now contains the following test components:
    1. InvoiceProcessor (AGENT, finance domain) - Invoice data extraction
    2. ContractAnalyzer (AGENT, legal domain) - Contract term analysis
    3. DocumentClassifier (TOOL, document_processing domain) - Document type classification
    4. CustomerFeedbackAnalyzer (AGENT, customer_service domain) - Sentiment analysis
    5. FinancialReportGenerator (TOOL, finance domain) - Financial report creation
    
    For each of these search queries, explain how the enhanced search would provide better results than the original:
    
    1. "financial data extraction from invoices"
    2. "legal document analysis with entity extraction"
    3. "document processing"
    
    Also explain:
    1. How would you instantiate and use the EnhancedSmartLibrary in a real application?
    2. What other components in the Evolving Agents Toolkit could benefit from similar evolution?
    3. How could the toolkit be enhanced to automate this kind of component evolution?
    
    Provide a comprehensive technical analysis.
    """
    
    comparison_result = await system_agent.run(comparison_prompt)
    logger.info("Search comparison analysis complete.")
    
    # Save the final results
    result = {
        "original_analysis": analysis_result.result.text,
        "evolution_result": evolution_result.result.text,
        "comparison_analysis": comparison_result.result.text,
        "evolved_component_path": "enhanced_smart_library.py"
    }
    
    with open("self_improvement_results.json", "w") as f:
        json.dump(result, f, indent=2)
    
    logger.info("\nSelf-improvement demonstration complete! Results saved to self_improvement_results.json")
    return result

if __name__ == "__main__":
    asyncio.run(evolve_smart_library())