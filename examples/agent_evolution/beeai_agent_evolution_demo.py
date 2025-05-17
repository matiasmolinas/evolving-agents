import asyncio
import logging
import os
import sys
import json
import time
import re
from typing import Dict, Any, List, Optional, Union, Tuple 
from dotenv import load_dotenv 

# Add parent directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))


from evolving_agents.smart_library.smart_library import SmartLibrary
from evolving_agents.core.llm_service import LLMService
from evolving_agents.core.system_agent import SystemAgentFactory
from evolving_agents.agent_bus.smart_agent_bus import SmartAgentBus
from evolving_agents.providers.registry import ProviderRegistry
from evolving_agents.providers.beeai_provider import BeeAIProvider
from evolving_agents.core.dependency_container import DependencyContainer
from evolving_agents.firmware.firmware import Firmware
from evolving_agents.core.mongodb_client import MongoDBClient 
from evolving_agents import config as eat_config 
from beeai_framework.agents.react import ReActAgent 

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables from .env file
if not load_dotenv():
    logger.warning(".env file not found or python-dotenv not installed. Environment variables might not be loaded from .env.")


# --- Helper: Ensure BeeAI Templates Exist ---
def ensure_templates_exist():
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    templates_dir = os.path.join(base_dir, "evolving_agents", "templates")
    os.makedirs(templates_dir, exist_ok=True)
    
    agent_template_path = os.path.join(templates_dir, "beeai_agent_template.txt")
    if not os.path.exists(agent_template_path):
        agent_template = """from typing import List, Dict, Any, Optional
import logging

from beeai_framework.agents.react import ReActAgent
from beeai_framework.agents.types import AgentMeta
from beeai_framework.memory import TokenMemory
from beeai_framework.backend.chat import ChatModel
from beeai_framework.tools.tool import Tool

class WeatherAgentInitializer:
    \"\"\"
    A specialized agent for providing weather information and forecasts.
    This agent can retrieve current weather, forecasts, and provide recommendations.
    \"\"\"
    
    @staticmethod
    def create_agent(
        llm: ChatModel, 
        tools: Optional[List[Tool]] = None,
        memory_type: str = "token"
    ) -> ReActAgent:
        \"\"\"
        Create and configure the weather agent.
        
        Args:
            llm: The language model to use
            tools: Optional list of tools to use
            memory_type: Type of memory to use
            
        Returns:
            Configured ReActAgent instance
        \"\"\"
        if tools is None:
            tools = []
            
        meta = AgentMeta(
            name="WeatherAgent",
            description=(
                "I am a weather expert that can provide current conditions, "
                "forecasts, and weather-related recommendations."
            ),
            tools=tools
        )
        
        memory = TokenMemory(llm)
        
        agent = ReActAgent(
            llm=llm,
            tools=tools,
            memory=memory,
            meta=meta
        )
        
        return agent"""
        
        with open(agent_template_path, 'w') as f:
            f.write(agent_template)
            
    tool_template_path = os.path.join(templates_dir, "beeai_tool_template.txt")
    if not os.path.exists(tool_template_path):
        tool_template = """from typing import Dict, Any, Optional
from pydantic import BaseModel, Field

from beeai_framework.tools.tool import Tool, StringToolOutput
from beeai_framework.context import RunContext
from beeai_framework.emitter.emitter import Emitter

class WeatherToolInput(BaseModel):
    \"\"\"Input schema for WeatherTool.\"\"\"
    location: str = Field(description="City or location to get weather for")
    units: str = Field(default="metric", description="Units to use (metric/imperial)")

class WeatherTool(Tool[WeatherToolInput, None, StringToolOutput]):
    \"\"\"Tool for retrieving current weather conditions and forecasts.\"\"\"
    
    name = "WeatherTool"
    description = "Get current weather conditions and forecasts for a location"
    input_schema = WeatherToolInput
    
    def __init__(self, api_key: Optional[str] = None, options: Optional[Dict[str, Any]] = None):
        super().__init__(options=options or {})
        self.api_key = api_key
        
    def _create_emitter(self) -> Emitter:
        return Emitter.root().child(
            namespace=["tool", "weather", "conditions"],
            creator=self,
        )
    
    async def _run(
        self, 
        tool_input: WeatherToolInput, 
        options: Optional[Dict[str, Any]] = None, 
        context: Optional[RunContext] = None
    ) -> StringToolOutput:
        \"\"\"
        Get weather for the specified location.
        \"\"\"
        try:
            weather_data = self._get_mock_weather(tool_input.location, tool_input.units)
            
            return StringToolOutput(
                f"Weather for {tool_input.location}:\\n"
                f"Temperature: {weather_data['temp']}°{'C' if tool_input.units == 'metric' else 'F'}\\n"
                f"Condition: {weather_data['condition']}\\n"
                f"Humidity: {weather_data['humidity']}%"
            )
        except Exception as e:
            return StringToolOutput(f"Error retrieving weather: {str(e)}")
            
    def _get_mock_weather(self, location: str, units: str) -> Dict[str, Any]:
        \"\"\"Mock weather data for demonstration.\"\"\"
        return {
            "temp": 22 if units == "metric" else 72,
            "condition": "Partly Cloudy",
            "humidity": 65
        }"""
        
        with open(tool_template_path, 'w') as f:
            f.write(tool_template)
    
    print("✓ BeeAI templates verified")

# --- AgentEvolutionTracker ---
class AgentEvolutionTracker:
    def __init__(self, storage_path="beeai_evolution_tracker.json"):
        self.storage_path = storage_path
        self.evolutions: Dict[str, List[Dict[str, Any]]] = {} 
        self._load_evolutions()
    
    def _load_evolutions(self):
        try:
            if os.path.exists(self.storage_path):
                with open(self.storage_path, 'r') as f:
                    self.evolutions = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            self.evolutions = {} 
    
    def _save_evolutions(self):
        with open(self.storage_path, 'w') as f:
            json.dump(self.evolutions, f, indent=2)
    
    def record_evolution(self, original_id: str, evolved_id: str, strategy: str, notes: str):
        if original_id not in self.evolutions:
            self.evolutions[original_id] = []
        
        self.evolutions[original_id].append({
            "evolved_id": evolved_id,
            "strategy": strategy,
            "notes": notes,
            "timestamp": time.time() 
        })
        self._save_evolutions()
    
    def get_evolution_history(self, agent_id: str) -> List[Dict[str, Any]]:
        return self.evolutions.get(agent_id, [])

# --- Helper: extract_component_id ---
def extract_component_id(response_text: str, id_field: str = "id") -> Optional[str]:
    if not response_text: return None
    try:
        data = json.loads(response_text)
        if isinstance(data, dict):
            if "results" in data and isinstance(data["results"], list) and data["results"]:
                return data["results"][0].get(id_field)
            elif "record_id" in data: 
                return data["record_id"]
            elif "evolved_id" in data: 
                return data["evolved_id"]
            elif "plan_id" in data: 
                return data["plan_id"]
            elif id_field in data: 
                return data[id_field]
    except json.JSONDecodeError:
        logger.debug(f"Failed to parse response_text as JSON for ID extraction: {response_text[:200]}")

    id_patterns = [
        rf'"record_id":\s*"([^"]+)"',
        rf'"evolved_id":\s*"([^"]+)"',
        rf'"plan_id":\s*"([^"]+)"',
        rf'"{id_field}":\s*"([^"]+)"', 
        r'ID:\s*([a-f0-9]{8}-(?:[a-f0-9]{4}-){3}[a-f0-9]{12})', 
        r'([a-f0-9]{8}-(?:[a-f0-9]{4}-){3}[a-f0-9]{12})' 
    ]
    for pattern in id_patterns:
        match = re.search(pattern, response_text, re.IGNORECASE)
        if match:
            return match.group(1)
    logger.warning(f"Could not extract component ID using field '{id_field}' or regex from: {response_text[:200]}")
    return None

# --- Sample Datasets ---
SAMPLE_WEATHER_QUERY = "What's the weather like in New York today?"
SAMPLE_FINANCIAL_DATA = """
Financial Summary: Q4 2023
Revenue: $10.2M
Expenses: $7.8M
Net Profit: $2.4M
Growth Rate: 18%
Key Products:
1. Product A: $4.5M (44%)
2. Product B: $3.2M (31%)
3. Product C: $2.5M (25%)
"""
SAMPLE_CUSTOMER_FEEDBACK = """
Customer: John D.
Rating: 3/5
Comments: The product works well for basic tasks, but I found the advanced features 
confusing to use. The interface could be more intuitive. Customer service was quick 
to respond when I had questions, which was helpful. Would recommend for beginners but 
not for power users looking for more sophisticated functionality.
"""
SAMPLE_WEBSITE_CONTENT = """
<html>
<head><title>Company Products</title></head>
<body>
  <h1>Our Premium Products</h1>
  <div class="product">
    <h2>Product A</h2>
    <p>Price: $199.99</p>
    <p>Rating: 4.7/5</p>
    <p>Description: Our flagship product with advanced features.</p>
  </div>
  <div class="product">
    <h2>Product B</h2>
    <p>Price: $149.99</p>
    <p>Rating: 4.5/5</p>
    <p>Description: Best value for budget-conscious customers.</p>
  </div>
</body>
</html>
"""
SAMPLE_FINANCIAL_FEEDBACK = """
I've been using your banking app for 3 months now. The interest rates are competitive,
but the monthly fees are too high compared to other banks. The customer service
responded quickly when I had an issue with a transfer, but your mortgage application
process is too complicated and took way too long. I like the investment options though.
"""

async def create_sentiment_analysis_tool(system_agent: ReActAgent) -> Tuple[Optional[str], Any]:
    """Create a sentiment analysis tool using the System Agent"""
    print("\n" + "-"*80)
    print("CREATING SENTIMENT ANALYSIS TOOL (BeeAI)")
    print("-"*80)
    
    create_tool_prompt = """
    I need a new sentiment analysis tool for analyzing text. Please create it using the BeeAI framework.
    
    Name: BeeAISentimentAnalysisTool
    Domain: text_analysis
    Description: BeeAI Tool for analyzing sentiment (positive, negative, neutral) in text and providing a score.
    
    The tool should:
    1. Accept text input.
    2. Determine sentiment: positive, negative, or neutral.
    3. Provide a sentiment score from -1.0 (very negative) to 1.0 (very positive).
    4. Return results as a structured JSON string.
    
    Ensure it's a complete, executable BeeAI Tool class.
    """
    
    print("\nCreating BeeAISentimentAnalysisTool...")
    create_result = await system_agent.run(create_tool_prompt)
    
    print("\nTool creation result:")
    print(create_result.result.text)
    
    tool_id = extract_component_id(create_result.result.text, "record_id")
    if tool_id:
        print(f"✓ Sentiment tool created with ID: {tool_id}")
    else:
        print("✗ Failed to extract sentiment tool ID from creation response.")
    return tool_id, create_result

async def create_customer_feedback_agent(system_agent: ReActAgent, sentiment_tool_id: Optional[str] = None) -> Tuple[Optional[str], Any]:
    """Create a customer feedback analysis agent using the System Agent"""
    print("\n" + "-"*80)
    print("CREATING CUSTOMER FEEDBACK AGENT (BeeAI)")
    print("-"*80)
    
    tool_integration_instruction = ""
    if sentiment_tool_id:
        tool_integration_instruction = f"This agent should be configured to use the 'BeeAISentimentAnalysisTool' (ID: {sentiment_tool_id}) if appropriate for its tasks."
    
    feedback_agent_prompt = f"""
    I need a new agent for analyzing customer feedback, built with the BeeAI framework.
    
    Name: BeeAICustomerFeedbackAgent
    Domain: customer_service
    Description: BeeAI Agent that analyzes customer feedback to extract insights, sentiment, and suggestions.
    
    The agent should:
    1. Extract key complaints and praise points.
    2. Identify product features mentioned.
    3. Categorize feedback by sentiment and topic.
    4. Provide actionable recommendations.
    {tool_integration_instruction}
    
    Implement a complete BeeAI Agent class.
    """
    
    print("\nCreating BeeAICustomerFeedbackAgent...")
    agent_result = await system_agent.run(feedback_agent_prompt)
    
    print("\nAgent creation result:")
    print(agent_result.result.text)
    
    agent_id = extract_component_id(agent_result.result.text, "record_id")
    if agent_id:
        print(f"✓ Customer feedback agent created with ID: {agent_id}")
    else:
        print("✗ Failed to extract customer feedback agent ID from creation response.")
    return agent_id, agent_result

async def evolve_sentiment_tool(system_agent: ReActAgent, tool_id: Optional[str] = None) -> Tuple[Optional[str], Any]:
    """Evolve the sentiment analysis tool"""
    print("\n" + "-"*80)
    print("EVOLVING SENTIMENT ANALYSIS TOOL (BeeAI)")
    print("-"*80)
    
    tool_to_evolve_name = "BeeAISentimentAnalysisTool" 
    
    evolve_prompt = f"""
    I want to evolve the '{tool_to_evolve_name}' tool to enhance its capabilities.
    If its ID is '{tool_id}', use that directly. Otherwise, search for it by name.
    
    The enhanced version should:
    1. Support multiple languages (at least English, Spanish, French).
    2. Detect specific emotions (joy, anger, sadness, surprise) in addition to general sentiment.
    3. Provide confidence scores for detected emotions.
    
    Use a standard evolution strategy.
    """
    
    print(f"\nEvolving {tool_to_evolve_name}...")
    evolve_result = await system_agent.run(evolve_prompt)
    
    print("\nEvolution result:")
    print(evolve_result.result.text)
    
    evolved_tool_id = extract_component_id(evolve_result.result.text, "evolved_id")
    if evolved_tool_id:
        print(f"✓ Sentiment tool evolved. New ID: {evolved_tool_id}")
    else:
        print(f"✗ Failed to extract evolved sentiment tool ID. Original ID was: {tool_id}")
        evolved_tool_id = tool_id 
        
    return evolved_tool_id, evolve_result

async def adapt_to_finance_domain(system_agent: ReActAgent, agent_id: Optional[str] = None) -> Tuple[Optional[str], Any]:
    """Adapt the customer feedback agent to the financial domain"""
    print("\n" + "-"*80)
    print("ADAPTING AGENT TO FINANCIAL DOMAIN (BeeAI)")
    print("-"*80)
    
    agent_to_adapt_name = "BeeAICustomerFeedbackAgent"
    
    adapt_prompt = f"""
    I want to adapt our '{agent_to_adapt_name}' (ID: {agent_id if agent_id else 'search by name'}) for the financial domain.
    
    The adapted agent, to be named 'BeeAIFinancialFeedbackAgent', should:
    1. Extract financial-specific metrics from feedback (e.g., mentions of interest rates, fees).
    2. Identify mentions of financial products (e.g., loans, investments).
    3. Flag potential compliance-related issues or complaints.
    
    Use a domain_adaptation strategy.
    """
    
    print(f"\nAdapting {agent_to_adapt_name} to financial domain...")
    adapt_result = await system_agent.run(adapt_prompt)
    
    print("\nDomain adaptation result:")
    print(adapt_result.result.text)
    
    adapted_agent_id = extract_component_id(adapt_result.result.text, "evolved_id")
    if adapted_agent_id:
        print(f"✓ Agent adapted to finance domain. New ID: {adapted_agent_id}")
    else:
        print(f"✗ Failed to extract adapted financial agent ID. Original ID was: {agent_id}")
        adapted_agent_id = agent_id 
        
    return adapted_agent_id, adapt_result

async def transform_to_web_analyzer(system_agent: ReActAgent, tool_id: Optional[str] = None) -> Tuple[Optional[str], Any]:
    """Aggressively evolve a tool into a web content analyzer"""
    print("\n" + "-"*80)
    print("TRANSFORMING TOOL TO WEB CONTENT ANALYZER (BeeAI)")
    print("-"*80)
    
    tool_to_transform_identifier = f"tool with ID '{tool_id}'" if tool_id else "the evolved sentiment analysis tool"

    transform_prompt = f"""
    I want to aggressively transform {tool_to_transform_identifier} into a comprehensive web content analyzer.
    The new tool should be named 'BeeAIWebContentAnalyzer'.
    
    It should:
    1. Extract main topics and themes from HTML web content.
    2. Identify key entities (people, organizations, locations).
    3. Extract product information if present (name, price, rating).
    4. Generate a concise summary of the web page content.
    
    This requires an aggressive evolution strategy.
    """
    
    print(f"\nAggressively transforming {tool_to_transform_identifier} into web content analyzer...")
    transform_result = await system_agent.run(transform_prompt)
    
    print("\nTransformation result:")
    print(transform_result.result.text)
    
    transformed_tool_id = extract_component_id(transform_result.result.text, "evolved_id")
    if transformed_tool_id:
        print(f"✓ Tool transformed to web analyzer. New ID: {transformed_tool_id}")
    else:
        print(f"✗ Failed to extract transformed web analyzer ID. Original ID was: {tool_id}")
        transformed_tool_id = tool_id 
        
    return transformed_tool_id, transform_result

async def test_evolved_components(system_agent: ReActAgent, smart_library: SmartLibrary, component_ids: Dict[str, Optional[str]]):
    """Test the evolved components"""
    print("\n" + "-"*80)
    print("TESTING EVOLVED BEEAI COMPONENTS")
    print("-"*80)
    
    evolved_sentiment_tool_id = component_ids.get("evolved_sentiment_tool")
    if evolved_sentiment_tool_id:
        print(f"\nTesting Evolved Sentiment Tool (ID: {evolved_sentiment_tool_id})...")
        sentiment_test_prompt = f"""
        Use the evolved sentiment analysis tool (ID: {evolved_sentiment_tool_id}) to analyze these texts for sentiment and emotions:
        1. English: "I am overjoyed with the new features! It's fantastic!"
        2. Spanish: "Estoy muy decepcionado con el producto, no funciona como esperaba."
        Provide structured results for each.
        """
        sentiment_result = await system_agent.run(sentiment_test_prompt)
        print("\nEvolved Sentiment Analysis Results:")
        print(sentiment_result.result.text)

    financial_agent_id = component_ids.get("financial_agent")
    if financial_agent_id:
        print(f"\nTesting Financial Feedback Agent (ID: {financial_agent_id})...")
        financial_test_prompt = f"""
        Use the 'BeeAIFinancialFeedbackAgent' (ID: {financial_agent_id}) to analyze this financial customer feedback:
        "{SAMPLE_FINANCIAL_FEEDBACK}"
        Extract financial metrics, product mentions, compliance flags, and sentiment.
        """
        financial_result = await system_agent.run(financial_test_prompt)
        print("\nFinancial Feedback Analysis Results:")
        print(financial_result.result.text)

    web_analyzer_id = component_ids.get("web_analyzer")
    if web_analyzer_id:
        print(f"\nTesting Web Content Analyzer (ID: {web_analyzer_id})...")
        web_test_prompt = f"""
        Use the 'BeeAIWebContentAnalyzer' (ID: {web_analyzer_id}) to analyze this HTML content:
        ```html
        {SAMPLE_WEBSITE_CONTENT}
        ```
        Extract topics, entities, product info, and summarize.
        """
        web_result = await system_agent.run(web_test_prompt)
        print("\nWeb Content Analysis Results:")
        print(web_result.result.text)

async def compare_components(system_agent: ReActAgent, smart_library: SmartLibrary, original_id: Optional[str], evolved_id: Optional[str]):
    """Compare original and evolved components"""
    if not original_id or not evolved_id:
        print("Skipping comparison: Missing original or evolved component ID.")
        return
        
    print("\n" + "-"*80)
    print(f"COMPARING Original ID: {original_id} vs. Evolved ID: {evolved_id}")
    print("-"*80)
    
    comparison_prompt = f"""
    Please compare two versions of a component from our SmartLibrary:
    Original Component ID: {original_id}
    Evolved Component ID: {evolved_id}
    
    Analyze:
    1. Differences in capabilities and descriptions.
    2. How the evolution addressed any specified requirements (if known).
    3. Potential improvements in the evolved version.
    Provide a summary of the comparison.
    """
    
    print("\nComparing components...")
    comparison_result = await system_agent.run(comparison_prompt)
    
    print("\nComparison Results:")
    print(comparison_result.result.text)

async def analyze_library_health(system_agent: ReActAgent, smart_library: SmartLibrary):
    """Analyze the health and structure of the library"""
    print("\n" + "-"*80)
    print("LIBRARY HEALTH ANALYSIS")
    print("-"*80)
    
    library_status = await smart_library.get_status_summary()
    
    analysis_prompt = f"""
    Please analyze the health and structure of our SmartLibrary based on these current stats:
    Total Records: {library_status.get('total_records', 0)}
    Components by Type: {json.dumps(library_status.get('by_type', {}), indent=2)}
    Domains: {', '.join(library_status.get('domains', []))}
    Active/Inactive Status: {json.dumps(library_status.get('by_status', {}), indent=2)}
    
    Provide an assessment covering:
    1. Distribution of components by type and domain.
    2. Identification of potential gaps or redundancies.
    3. Suggestions for organizational improvements or new components.
    """
    
    print("\nAnalyzing library health...")
    analysis_result = await system_agent.run(analysis_prompt)
    
    print("\nLibrary Health Analysis:")
    print(analysis_result.result.text)

async def setup_and_clean_demo_environment(container: DependencyContainer):
    """Sets up the MongoDB environment and cleans specific demo data."""
    print("\n[MONGODB DEMO SETUP & CLEANUP]")
    
    smart_library: SmartLibrary = container.get('smart_library')
    if not smart_library or smart_library.components_collection is None: # Corrected check
        logger.error("SmartLibrary could not be initialized with MongoDB or components_collection is None. Demo cannot proceed with DB cleanup.")
        return

    demo_domains = ["text_analysis", "customer_service", "finance_feedback_beeai", "web_content_beeai"]
    demo_names_to_delete = [
        "BeeAISentimentAnalysisTool", 
        "BeeAICustomerFeedbackAgent", 
        "BeeAIFinancialFeedbackAgent", 
        "BeeAIWebContentAnalyzer",
    ]

    print(f"  Targeting demo domains for cleanup: {demo_domains}")
    print(f"  Targeting demo names for cleanup: {demo_names_to_delete}")

    delete_count = 0
    # The collection is already confirmed not to be None by the check above
    for name in demo_names_to_delete:
        records_to_delete = await smart_library.components_collection.find({"name": name}).to_list(length=None)
        if records_to_delete:
            ids_to_delete = [r["id"] for r in records_to_delete]
            result = await smart_library.components_collection.delete_many({"id": {"$in": ids_to_delete}})
            delete_count += result.deleted_count
            print(f"    Deleted {result.deleted_count} records with name '{name}'.")

    for domain in demo_domains:
        result = await smart_library.components_collection.delete_many({"domain": domain})
        delete_count += result.deleted_count
        print(f"    Deleted {result.deleted_count} records from domain '{domain}'.")

    print(f"  MongoDB: Cleaned up {delete_count} previous BeeAI evolution demo components.")

    tracker_path = "beeai_evolution_tracker.json"
    if os.path.exists(tracker_path):
        os.remove(tracker_path)
        print(f"  Local Tracker: Deleted '{tracker_path}'.")
    
    print("✓ Demo environment setup/cleanup complete.")


async def main():
    try:
        print("\n" + "="*80)
        print("BEEAI AGENT EVOLUTION DEMONSTRATION (MongoDB Backend)")
        print("="*80)
        
        ensure_templates_exist()
        
        container = DependencyContainer()
        
        try:
            mongo_client = MongoDBClient() 
            await mongo_client.ping_server() 
            container.register('mongodb_client', mongo_client)
            print(f"✓ MongoDB Client initialized and connected to DB: '{mongo_client.db_name}'")
        except Exception as e:
            logger.error(f"CRITICAL: Failed to initialize MongoDBClient: {e}. Ensure .env is configured and MongoDB is accessible.")
            return

        llm_service = LLMService(provider="openai", model="gpt-4o", container=container) 
        container.register('llm_service', llm_service)
        
        smart_library = SmartLibrary(container=container)
        container.register('smart_library', smart_library)
        
        firmware = Firmware()
        container.register('firmware', firmware)
        
        agent_bus = SmartAgentBus(container=container)
        container.register('agent_bus', agent_bus)
        
        provider_registry = ProviderRegistry()
        provider_registry.register_provider(BeeAIProvider(llm_service)) 
        container.register('provider_registry', provider_registry)
        
        print("\nInitializing System Agent...")
        system_agent = await SystemAgentFactory.create_agent(container=container)
        container.register('system_agent', system_agent)
        
        await setup_and_clean_demo_environment(container)

        await smart_library.initialize() 
        await agent_bus.initialize_from_library() 
        
        evolution_tracker = AgentEvolutionTracker()
        component_ids: Dict[str, Optional[str]] = {} 

        print("\n[PHASE 1: COMPONENT CREATION]")
        sentiment_tool_id, _ = await create_sentiment_analysis_tool(system_agent)
        component_ids["original_sentiment_tool"] = sentiment_tool_id
        await asyncio.sleep(1) 
        
        agent_id, _ = await create_customer_feedback_agent(system_agent, sentiment_tool_id)
        component_ids["original_feedback_agent"] = agent_id
        await asyncio.sleep(1)
        
        print("\n[PHASE 2: STANDARD EVOLUTION OF SENTIMENT TOOL]")
        evolved_tool_id, _ = await evolve_sentiment_tool(system_agent, sentiment_tool_id)
        component_ids["evolved_sentiment_tool"] = evolved_tool_id
        if sentiment_tool_id and evolved_tool_id and sentiment_tool_id != evolved_tool_id:
            evolution_tracker.record_evolution(sentiment_tool_id, evolved_tool_id, "standard", "Added multi-language and emotion detection")
        await asyncio.sleep(1)

        print("\n[PHASE 3: DOMAIN ADAPTATION OF FEEDBACK AGENT]")
        financial_agent_id, _ = await adapt_to_finance_domain(system_agent, agent_id)
        component_ids["financial_agent"] = financial_agent_id
        if agent_id and financial_agent_id and agent_id != financial_agent_id:
            evolution_tracker.record_evolution(agent_id, financial_agent_id, "domain_adaptation", "Adapted to financial domain")
        await asyncio.sleep(1)

        print("\n[PHASE 4: AGGRESSIVE EVOLUTION (TRANSFORMATION)]")
        web_analyzer_id, _ = await transform_to_web_analyzer(system_agent, evolved_tool_id) 
        component_ids["web_analyzer"] = web_analyzer_id
        if evolved_tool_id and web_analyzer_id and evolved_tool_id != web_analyzer_id:
            evolution_tracker.record_evolution(evolved_tool_id, web_analyzer_id, "aggressive", "Transformed to WebContentAnalyzer")
        await asyncio.sleep(1)
        
        print("\n[PHASE 5: TESTING EVOLVED COMPONENTS]")
        await test_evolved_components(system_agent, smart_library, component_ids)
        
        print("\n[PHASE 6: COMPONENT COMPARISON]")
        await compare_components(system_agent, smart_library, 
                                 component_ids.get("original_sentiment_tool"), 
                                 component_ids.get("evolved_sentiment_tool"))
        
        print("\n[PHASE 7: LIBRARY HEALTH ANALYSIS]")
        await analyze_library_health(system_agent, smart_library)
        
        print("\n" + "="*80)
        print("BEEAI AGENT EVOLUTION DEMONSTRATION (MongoDB Backend) COMPLETED")
        print("="*80)
        
    except Exception as e:
        logger.error(f"Error in BeeAI Evolution Demo: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    if not os.getenv("MONGODB_URI") or not os.getenv("OPENAI_API_KEY"):
        logger.error("CRITICAL: MONGODB_URI and/or OPENAI_API_KEY not found in .env file or environment.")
        logger.error("Please ensure your .env file is correctly configured and python-dotenv is installed.")
    else:
        asyncio.run(main())