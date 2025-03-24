import asyncio
import pytest
import pytest_asyncio
import tempfile
import os
from datetime import datetime

from evolving_agents.core.llm_service import LLMService
from evolving_agents.smart_library.smart_library import SmartLibrary

@pytest_asyncio.fixture
async def smart_library():
    """
    Create temporary storage and vector DB directories to avoid polluting the workspace.
    This fixture instantiates the real LLM service (ensure your API keys are set) and the SmartLibrary.
    """
    temp_dir = tempfile.TemporaryDirectory()
    storage_path = os.path.join(temp_dir.name, "smart_library.json")
    vector_db_path = os.path.join(temp_dir.name, "vector_db")
    
    # Instantiate the real LLM service.
    llm_service = LLMService(
        provider="openai",
        model="gpt-4o",
        embedding_model="text-embedding-3-small"
    )
    library = SmartLibrary(
        storage_path=storage_path,
        vector_db_path=vector_db_path,
        llm_service=llm_service
    )
    # Allow time for asynchronous vector DB sync tasks to initiate.
    await asyncio.sleep(1)
    yield library
    temp_dir.cleanup()

@pytest.mark.asyncio
async def test_create_and_find_record(smart_library):
    # A very long and detailed description for a document analyzer tool.
    long_description = (
        "This is a highly detailed and comprehensive Document Analyzer tool. "
        "Its functional requirements include parsing diverse document types such as invoices, contracts, and receipts; "
        "performing advanced text analysis to extract invoice numbers, vendor details, and numerical totals; "
        "ensuring robust error handling and scalability; integrating with external logging databases; "
        "and providing structured output along with detailed process logging. "
        "The agent is designed for self-monitoring and adaptive processing based on document complexity, "
        "making it ideal for enterprise-level document management systems."
    )
    record = await smart_library.create_record(
        name="LongDocumentAnalyzer",
        record_type="TOOL",
        domain="document_processing",
        description=long_description,
        code_snippet="# Implementation for Document Analyzer",
        version="1.0.0",
        tags=["document", "analyzer"]
    )
    # Verify that the record was created and stored.
    assert "id" in record
    found = await smart_library.find_record_by_id(record["id"])
    assert found is not None
    assert found["name"] == "LongDocumentAnalyzer"

@pytest.mark.asyncio
async def test_semantic_search(smart_library):
    # Create two records with long, detailed descriptions that define clear functional requirements.
    long_description_doc = (
        "This Document Analyzer tool is engineered for in-depth analysis of textual documents. "
        "It is capable of extracting invoice numbers, vendor names, dates, and document types with exceptional accuracy. "
        "It supports multiple file formats, includes dynamic error handling, and logs detailed processing steps for auditing. "
        "Additional functionalities include adaptive parsing and integration with external data sources to improve accuracy."
    )
    long_description_invoice = (
        "The Advanced Invoice Processor agent is designed for complex financial document processing. "
        "It must extract detailed financial data such as invoice numbers, dates, vendor information, line-item details, subtotals, taxes, and totals; "
        "verify the accuracy of numerical calculations; and generate a comprehensive structured summary. "
        "The agent includes robust error detection, supports multiple invoice formats, and integrates seamlessly with financial reporting systems. "
        "Furthermore, it is optimized for real-time processing and provides extensive logging for audit purposes."
    )
    record_doc = await smart_library.create_record(
        name="LongDocumentAnalyzer",
        record_type="TOOL",
        domain="document_processing",
        description=long_description_doc,
        code_snippet="# Code for Document Analyzer",
        version="1.0.0",
        tags=["document", "analyzer"]
    )
    record_invoice = await smart_library.create_record(
        name="AdvancedInvoiceProcessor",
        record_type="AGENT",
        domain="financial_processing",
        description=long_description_invoice,
        code_snippet="# Code for Invoice Processor",
        version="1.0.0",
        tags=["invoice", "processor", "advanced"]
    )
    # Allow time for the vector DB to sync records.
    await asyncio.sleep(2)
    
    # Search using a query that closely resembles the requirements of invoice processing.
    query = (
        "Extract and verify detailed financial data from invoices with comprehensive error detection, "
        "robust reporting, and real-time processing capabilities."
    )
    results = await smart_library.semantic_search(
        query=query,
        record_type="AGENT",
        limit=2,
        threshold=0.1
    )
    
    # Check that the AdvancedInvoiceProcessor is among the search results.
    found_invoice = any(r["id"] == record_invoice["id"] for r, sim in results)
    assert found_invoice, "AdvancedInvoiceProcessor should be returned by semantic search."
    
    # Simulate increased usage for the invoice processor to test the boost effect.
    for _ in range(50):
        await smart_library.update_usage_metrics(record_invoice["id"], success=True)
    await asyncio.sleep(1)
    
    results_after = await smart_library.semantic_search(
        query=query,
        record_type="AGENT",
        limit=2,
        threshold=0.1
    )
    sim_before = next((sim for r, sim in results if r["id"] == record_invoice["id"]), None)
    sim_after = next((sim for r, sim in results_after if r["id"] == record_invoice["id"]), None)
    # Check that the similarity score after usage updates is at least as high as before.
    if sim_before is not None:
        assert sim_after >= sim_before, "Usage metrics boost should increase the similarity score."

@pytest.mark.asyncio
async def test_search_by_tag(smart_library):
    # Create records with specific tags.
    record1 = await smart_library.create_record(
        name="TagTestAgent1",
        record_type="AGENT",
        domain="test_domain",
        description="Agent with functional capabilities for alpha and beta tasks in system orchestration.",
        code_snippet="# code",
        version="1.0.0",
        tags=["alpha", "beta"]
    )
    record2 = await smart_library.create_record(
        name="TagTestTool1",
        record_type="TOOL",
        domain="test_domain",
        description="Tool designed for gamma functionality in system integration.",
        code_snippet="# code",
        version="1.0.0",
        tags=["gamma"]
    )
    # Validate search by tag (including case-insensitive matching).
    results_alpha = await smart_library.search_by_tag(tags=["alpha"])
    assert any(r["id"] == record1["id"] for r in results_alpha)
    results_gamma = await smart_library.search_by_tag(tags=["gamma"])
    assert any(r["id"] == record2["id"] for r in results_gamma)
    results_beta = await smart_library.search_by_tag(tags=["BeTa"])
    assert any(r["id"] == record1["id"] for r in results_beta)

@pytest.mark.asyncio
async def test_update_usage_metrics(smart_library):
    # Create a record and then update its usage metrics.
    record = await smart_library.create_record(
        name="MetricsTestAgent",
        record_type="AGENT",
        domain="metrics",
        description="Agent to test proper updating of usage metrics. Must log each execution with success indicators.",
        code_snippet="# code",
        version="1.0.0",
        tags=["metrics"]
    )
    original_usage = record.get("usage_count", 0)
    await smart_library.update_usage_metrics(record["id"], success=True)
    updated_record = await smart_library.find_record_by_id(record["id"])
    assert updated_record["usage_count"] == original_usage + 1
    assert updated_record["success_count"] == 1

@pytest.mark.asyncio
async def test_evolve_record(smart_library):
    # Create an initial record with a long, detailed description.
    original_description = (
        "Original agent designed for processing financial documents. "
        "It must extract invoice details, validate numerical calculations, and generate structured summaries. "
        "Functional requirements also include robust error handling and comprehensive logging for audit purposes."
    )
    record = await smart_library.create_record(
        name="EvolveTestAgent",
        record_type="AGENT",
        domain="financial_processing",
        description=original_description,
        code_snippet="# original code",
        version="1.0.0",
        tags=["evolution", "test"]
    )
    new_code = "# evolved code with enhanced processing capabilities and improved error handling"
    evolved_record = await smart_library.evolve_record(
        parent_id=record["id"],
        new_code_snippet=new_code,
        description="Evolved agent with improved invoice processing, enhanced validation, and superior error handling."
    )
    # Validate that the evolved record correctly references its parent and has an incremented version.
    assert evolved_record["parent_id"] == record["id"]
    original_version = record["version"]
    new_version = evolved_record["version"]
    original_parts = original_version.split(".")
    new_parts = new_version.split(".")
    assert int(new_parts[-1]) == int(original_parts[-1]) + 1

