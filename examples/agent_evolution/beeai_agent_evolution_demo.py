import asyncio
import logging
import os
import sys
import json
import time
import re
from typing import Dict, Any, List, Optional, Union

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evolving_agents.smart_library.smart_library import SmartLibrary
from evolving_agents.core.llm_service import LLMService
from evolving_agents.core.system_agent import SystemAgentFactory
from evolving_agents.agent_bus.smart_agent_bus import SmartAgentBus
from evolving_agents.providers.registry import ProviderRegistry
from evolving_agents.providers.beeai_provider import BeeAIProvider
from evolving_agents.core.dependency_container import DependencyContainer
from evolving_agents.firmware.firmware import Firmware

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Sample datasets
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

# Define BeeAI templates to make sure they're available
def ensure_templates_exist():
    """Ensure that template files for BeeAI agents and tools exist"""
    templates_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "templates")
    os.makedirs(templates_dir, exist_ok=True)
    
    # BeeAI Agent Template
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
        # Default to empty tools list if none provided
        if tools is None:
            tools = []
            
        # Define agent metadata
        meta = AgentMeta(
            name="WeatherAgent",
            description=(
                "I am a weather expert that can provide current conditions, "
                "forecasts, and weather-related recommendations."
            ),
            tools=tools
        )
        
        # Create memory based on specified type
        memory = TokenMemory(llm)
        
        # Create the agent
        agent = ReActAgent(
            llm=llm,
            tools=tools,
            memory=memory,
            meta=meta
        )
        
        return agent"""
        
        with open(agent_template_path, 'w') as f:
            f.write(agent_template)
            
    # BeeAI Tool Template
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
        input: WeatherToolInput, 
        options: Optional[Dict[str, Any]] = None, 
        context: Optional[RunContext] = None
    ) -> StringToolOutput:
        \"\"\"
        Get weather for the specified location.
        
        Args:
            input: Input parameters with location and units
            
        Returns:
            Weather information as text
        \"\"\"
        try:
            # In a real implementation, this would call a weather API
            # This is just a mock implementation for demonstration purposes
            weather_data = self._get_mock_weather(input.location, input.units)
            
            return StringToolOutput(
                f"Weather for {input.location}:\\n"
                f"Temperature: {weather_data['temp']}°{input.units == 'metric' and 'C' or 'F'}\\n"
                f"Condition: {weather_data['condition']}\\n"
                f"Humidity: {weather_data['humidity']}%"
            )
        except Exception as e:
            return StringToolOutput(f"Error retrieving weather: {str(e)}")
            
    def _get_mock_weather(self, location: str, units: str) -> Dict[str, Any]:
        \"\"\"Mock weather data for demonstration.\"\"\"
        # In a real implementation, this would be replaced with API calls
        return {
            "temp": 22 if units == "metric" else 72,
            "condition": "Partly Cloudy",
            "humidity": 65
        }"""
        
        with open(tool_template_path, 'w') as f:
            f.write(tool_template)
    
    print("✓ BeeAI templates verified")

class AgentEvolutionTracker:
    """Tracks agent evolution for demonstration purposes"""
    def __init__(self, storage_path="beeai_evolution_tracker.json"):
        self.storage_path = storage_path
        self.evolutions = {}
        self._load_evolutions()
    
    def _load_evolutions(self):
        try:
            if os.path.exists(self.storage_path):
                with open(self.storage_path, 'r') as f:
                    self.evolutions = json.load(f)
        except:
            self.evolutions = {}
    
    def _save_evolutions(self):
        with open(self.storage_path, 'w') as f:
            json.dump(self.evolutions, f, indent=2)
    
    def record_evolution(self, original_id, evolved_id, strategy, notes):
        """Record an evolution event"""
        if original_id not in self.evolutions:
            self.evolutions[original_id] = []
        
        self.evolutions[original_id].append({
            "evolved_id": evolved_id,
            "strategy": strategy,
            "notes": notes,
            "timestamp": time.time()
        })
        
        self._save_evolutions()
    
    def get_evolution_history(self, agent_id):
        """Get evolution history for an agent"""
        return self.evolutions.get(agent_id, [])

# Helper function to extract component ID from JSON response or text
def extract_component_id(response_text, id_field="id"):
    """Extract a component ID from a response text, using multiple approaches"""
    # Try JSON parsing first
    try:
        data = json.loads(response_text)
        if isinstance(data, dict):
            # If it's a search result with multiple items
            if "results" in data and data["results"] and isinstance(data["results"], list):
                return data["results"][0].get(id_field, "")
            # If it's a single item result
            elif "status" in data and data["status"] == "success" and id_field in data:
                return data[id_field]
            elif "record_id" in data:
                return data["record_id"]
            elif "evolved_id" in data:
                return data["evolved_id"]
    except:
        pass
    
    # Try regex patterns for ID extraction
    id_patterns = [
        rf'record_id":\s*"([^"]+)"',  # Looking for "record_id": "abc123"
        rf'record id":\s*"([^"]+)"',  # Looking for "record id": "abc123"
        rf'"id":\s*"([^"]+)"',        # Looking for "id": "abc123"
        rf'id:\s*([a-f0-9-]+)',       # Looking for id: abc123
        r'([a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12})'  # UUID pattern
    ]
    
    for pattern in id_patterns:
        matches = re.findall(pattern, response_text, re.IGNORECASE)
        if matches:
            return matches[0]
    
    return None

async def create_sentiment_analysis_tool(system_agent):
    """Create a sentiment analysis tool using the System Agent"""
    print("\n" + "-"*80)
    print("CREATING SENTIMENT ANALYSIS TOOL")
    print("-"*80)
    
    create_tool_prompt = """
    I need a new sentiment analysis tool for analyzing text. Create a BeeAI tool with these specifications:
    
    Name: SentimentAnalysisTool
    Domain: text_analysis
    Description: Tool for analyzing sentiment in text using natural language processing
    
    The tool should:
    1. Accept text input and optional parameters for analysis depth
    2. Determine if sentiment is positive, negative, or neutral
    3. Provide a sentiment score on a scale from -1.0 (very negative) to 1.0 (very positive)
    4. Extract key sentiment-driving phrases from the text
    5. Handle errors gracefully
    6. Return results in a clear, structured format
    
    Use the BeeAI framework and implement a proper Tool class with input schema.
    """
    
    print("\nCreating SentimentAnalysisTool...")
    create_result = await system_agent.run(create_tool_prompt)
    
    print("\nTool creation result:")
    print(create_result.result.text)
    
    # Extract the tool ID from the response
    tool_id = extract_component_id(create_result.result.text, "record_id")
    
    # If ID wasn't found in the primary response, try to find it with a regex pattern for UUIDs
    if not tool_id:
        uuid_pattern = r'([a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12})'
        matches = re.findall(uuid_pattern, create_result.result.text)
        if matches:
            tool_id = matches[0]
    
    return tool_id, create_result

async def create_customer_feedback_agent(system_agent, sentiment_tool_id=None):
    """Create a customer feedback analysis agent using the System Agent"""
    print("\n" + "-"*80)
    print("CREATING CUSTOMER FEEDBACK AGENT")
    print("-"*80)
    
    feedback_agent_prompt = """
    I need a new agent for analyzing customer feedback. Create a BeeAI agent with these specifications:
    
    Name: CustomerFeedbackAgent
    Domain: customer_service
    Description: Agent that analyzes customer feedback to extract insights and suggestions
    
    The agent should:
    1. Extract key complaints and praise points from customer feedback
    2. Identify product features mentioned in feedback
    3. Categorize feedback by sentiment and topic
    4. Provide actionable recommendations based on the feedback
    5. If possible, integrate with any existing sentiment analysis tools
    
    Use the BeeAI framework and implement a proper Agent class with the required initialization.
    """
    
    print("\nCreating CustomerFeedbackAgent...")
    agent_result = await system_agent.run(feedback_agent_prompt)
    
    print("\nAgent creation result:")
    print(agent_result.result.text)
    
    # Extract the agent ID from the response
    agent_id = extract_component_id(agent_result.result.text, "record_id")
    
    # If ID wasn't found in the primary response, try to find it with a regex pattern for UUIDs
    if not agent_id:
        uuid_pattern = r'([a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12})'
        matches = re.findall(uuid_pattern, agent_result.result.text)
        if matches:
            agent_id = matches[0]
    
    return agent_id, agent_result

async def evolve_sentiment_tool(system_agent, tool_id=None):
    """Evolve the sentiment analysis tool to support multiple languages and emotions"""
    print("\n" + "-"*80)
    print("EVOLVING SENTIMENT ANALYSIS TOOL")
    print("-"*80)
    
    # First, search for the tool if ID not provided
    if not tool_id:
        search_prompt = """
        I need to find a sentiment analysis tool in our library. Please search for any tools
        in the text_analysis domain that can analyze sentiment in text.
        """
        
        print("\nSearching for sentiment analysis tool...")
        search_result = await system_agent.run(search_prompt)
        print("\nSearch result:")
        print(search_result.result.text)
        
        # Extract the tool ID from the search result
        tool_id = extract_component_id(search_result.result.text)
    
    # Now evolve the tool
    evolve_prompt = f"""
    I want to evolve our sentiment analysis tool to enhance its capabilities.
    
    The enhanced version should:
    1. Support multiple languages (at least English, Spanish, French)
    2. Detect specific emotions beyond positive/negative (joy, anger, fear, surprise, etc.)
    3. Provide confidence scores for the detected emotions
    4. Extract emotionally charged phrases
    5. Include a summary of the overall emotional tone
    
    Use a standard evolution strategy that preserves the core functionality while adding these new features.
    """
    
    print("\nEvolving sentiment analysis tool...")
    evolve_result = await system_agent.run(evolve_prompt)
    
    print("\nEvolution result:")
    print(evolve_result.result.text)
    
    # Extract the evolved tool ID
    evolved_tool_id = extract_component_id(evolve_result.result.text, "record_id")
    if not evolved_tool_id:
        evolved_tool_id = tool_id  # Fallback to original ID if no new ID is found
    
    return evolved_tool_id, evolve_result

async def adapt_to_finance_domain(system_agent, agent_id=None):
    """Adapt the customer feedback agent to the financial domain"""
    print("\n" + "-"*80)
    print("ADAPTING AGENT TO FINANCIAL DOMAIN")
    print("-"*80)
    
    # First, search for the agent if ID not provided
    if not agent_id:
        search_prompt = """
        I need to find a customer feedback analysis agent in our library. Please search for any agents
        in the customer_service domain that can analyze customer feedback.
        """
        
        print("\nSearching for customer feedback agent...")
        search_result = await system_agent.run(search_prompt)
        print("\nSearch result:")
        print(search_result.result.text)
        
        # Extract the agent ID from the search result
        agent_id = extract_component_id(search_result.result.text)
    
    # Now adapt the agent to the financial domain
    adapt_prompt = f"""
    I want to adapt our customer feedback analysis agent to the financial domain.
    
    The adapted financial feedback agent should:
    1. Extract financial-specific metrics from feedback
    2. Identify mentions of financial products and services
    3. Flag compliance-related issues in customer feedback
    4. Tag feedback related to fees, rates, and customer service
    5. Include financial sentiment analysis
    
    Create a new version adapted to the finance domain, while maintaining the core feedback analysis capabilities.
    The new agent should be named "FinancialFeedbackAgent".
    """
    
    print("\nAdapting agent to financial domain...")
    adapt_result = await system_agent.run(adapt_prompt)
    
    print("\nDomain adaptation result:")
    print(adapt_result.result.text)
    
    # Extract the adapted agent ID
    adapted_agent_id = extract_component_id(adapt_result.result.text, "record_id")
    if not adapted_agent_id:
        adapted_agent_id = agent_id  # Fallback to original ID if no new ID is found
    
    return adapted_agent_id, adapt_result

async def transform_to_web_analyzer(system_agent, tool_id=None):
    """Aggressively evolve the sentiment tool into a web content analyzer"""
    print("\n" + "-"*80)
    print("TRANSFORMING TO WEB CONTENT ANALYZER")
    print("-"*80)
    
    # Search for the sentiment tool if needed
    if not tool_id:
        search_prompt = """
        I need to find a text analysis tool to evolve. Please search for any sentiment
        analysis tools in our library that we can transform.
        """
        
        print("\nSearching for a tool to transform...")
        search_result = await system_agent.run(search_prompt)
        
        # Extract tool ID
        tool_id = extract_component_id(search_result.result.text)
    
    # Now transform the tool
    transform_prompt = f"""
    I want to aggressively transform a text analysis tool into a comprehensive web content analyzer.
    
    The new web content analyzer should:
    1. Extract main topics and themes from web pages
    2. Identify key entities (people, organizations, locations)
    3. Extract product information with prices and ratings
    4. Parse structured data like tables and lists
    5. Generate a summary of the content
    6. Extract metadata about the page
    
    This requires an aggressive evolution approach that significantly changes the functionality.
    The new tool should be named "WebContentAnalyzer".
    """
    
    print("\nAggressively transforming tool into web content analyzer...")
    transform_result = await system_agent.run(transform_prompt)
    
    print("\nTransformation result:")
    print(transform_result.result.text)
    
    # Extract the transformed tool ID
    transformed_tool_id = extract_component_id(transform_result.result.text, "record_id") 
    if not transformed_tool_id:
        transformed_tool_id = tool_id  # Fallback to original ID if no new ID is found
    
    return transformed_tool_id, transform_result

async def test_evolved_components(system_agent, library, component_ids):
    """Test the evolved components to demonstrate their capabilities"""
    print("\n" + "-"*80)
    print("TESTING EVOLVED COMPONENTS")
    print("-"*80)
    
    # First, check if we have valid component IDs and their types
    component_types = {}
    
    # Find any components we need to test if not explicitly provided
    if not any(component_ids.values()):
        components = await library.export_records()
        for component in components:
            if component["record_type"] == "TOOL" and "sentiment" in component["name"].lower():
                component_ids["sentiment_tool"] = component["id"]
                component_types["sentiment_tool"] = component
            elif component["record_type"] == "TOOL" and "web" in component["name"].lower():
                component_ids["web_analyzer"] = component["id"]
                component_types["web_analyzer"] = component
            elif component["record_type"] == "AGENT" and "financial" in component["name"].lower():
                component_ids["financial_agent"] = component["id"]
                component_types["financial_agent"] = component
    else:
        # Get component details for provided IDs
        for component_key, component_id in component_ids.items():
            if component_id:
                component = await library.find_record_by_id(component_id)
                if component:
                    component_types[component_key] = component
    
    # Test sentiment analysis with multiple languages
    if "sentiment_tool" in component_types:
        sentiment_component = component_types["sentiment_tool"]
        print(f"\nTesting sentiment analysis tool: {sentiment_component['name']}...")
        sentiment_test_prompt = f"""
        I want to test our sentiment analysis tool with multiple languages.
        
        Use the tool to analyze the sentiment and emotions in these texts:
        
        English: "I absolutely love this product! It exceeded all my expectations."
        Spanish: "El servicio al cliente fue terrible, nunca volveré a comprar aquí."
        French: "Le produit est bon mais le prix est un peu élevé pour ce que c'est."
        
        Perform a detailed analysis of each and show the results.
        """
        
        sentiment_result = await system_agent.run(sentiment_test_prompt)
        print("\nMultilingual sentiment analysis results:")
        print(sentiment_result.result.text)
    
    # Test financial feedback agent
    if "financial_agent" in component_types:
        financial_component = component_types["financial_agent"]
        print(f"\nTesting financial feedback agent: {financial_component['name']}...")
        financial_test_prompt = f"""
        I want to test our financial feedback analysis agent.
        
        Use the agent to analyze this customer feedback about a financial product:
        
        {SAMPLE_FINANCIAL_FEEDBACK}
        
        Perform a detailed analysis and provide your findings.
        """
        
        financial_result = await system_agent.run(financial_test_prompt)
        print("\nFinancial feedback analysis results:")
        print(financial_result.result.text)
    
    # Test web content analyzer
    if "web_analyzer" in component_types:
        web_component = component_types["web_analyzer"]
        print(f"\nTesting web content analyzer: {web_component['name']}...")
        web_test_prompt = f"""
        I want to test our web content analyzer tool.
        
        Use the tool to analyze this HTML content:
        
        {SAMPLE_WEBSITE_CONTENT}
        
        Extract and provide:
        - Main topics and themes
        - Products with their prices and ratings
        - Structured data elements
        - A concise summary of the page content
        """
        
        web_result = await system_agent.run(web_test_prompt)
        print("\nWeb content analysis results:")
        print(web_result.result.text)

async def compare_components(system_agent, library, original_id=None, evolved_id=None):
    """Compare original and evolved components"""
    print("\n" + "-"*80)
    print("COMPONENT COMPARISON")
    print("-"*80)
    
    # If IDs not provided, find components to compare
    if not original_id or not evolved_id:
        print("\nFinding components to compare...")
        components_prompt = """
        I need to compare original and evolved versions of components in our library.
        
        Please find:
        1. An original sentiment analysis tool (likely version 1.0.0)
        2. Its evolved version with enhanced capabilities
        
        Return both components and their details so we can perform a comparison.
        """
        
        components_result = await system_agent.run(components_prompt)
        print("\nComponents found:")
        print(components_result.result.text)
        
        # Try to extract IDs from the result
        try:
            # Get components from library and sort by version
            components = await library.export_records()
            sentiment_tools = [c for c in components if c["record_type"] == "TOOL" and "sentiment" in c["name"].lower()]
            if len(sentiment_tools) >= 2:
                sentiment_tools.sort(key=lambda x: x.get("version", "0.0.0"))
                original_id = sentiment_tools[0]["id"]
                evolved_id = sentiment_tools[1]["id"]
            elif len(sentiment_tools) == 1:
                # If we only have one tool, compare it with itself (not ideal but prevents failure)
                original_id = evolved_id = sentiment_tools[0]["id"]
        except:
            print("✗ Could not automatically identify components for comparison")
            return
    
    # Now perform the comparison
    comparison_prompt = """
    Please compare the original and evolved versions of our sentiment analysis component.
    
    For your comparison:
    1. Analyze the capabilities of both versions
    2. Identify specific improvements in the evolved version
    3. Evaluate how well the evolution addressed the enhancement requirements
    4. Suggest any further improvements that could be made
    5. Score both versions (1-10) on:
       - Functionality
       - Flexibility
       - Performance
       - Overall quality
    
    Provide a detailed analysis that highlights the evolution benefits and remaining opportunities.
    """
    
    print("\nComparing original and evolved components...")
    comparison_result = await system_agent.run(comparison_prompt)
    
    print("\nComparison results:")
    print(comparison_result.result.text)

async def analyze_library_health(system_agent, library):
    """Analyze the health and structure of the library"""
    print("\n" + "-"*80)
    print("LIBRARY HEALTH ANALYSIS")
    print("-"*80)
    
    # Get library status summary
    library_status = await library.get_status_summary()
    
    # Create a more explicit prompt with the library status data
    analysis_prompt = f"""
    Please analyze the health and structure of our SmartLibrary.
    
    Current Library Stats:
    - Total Records: {library_status.get('total_records', 0)}
    - Components by Type: {json.dumps(library_status.get('by_type', {}), indent=2)}
    - Domains: {', '.join(library_status.get('domains', []))}
    - Active/Inactive: {json.dumps(library_status.get('by_status', {}), indent=2)}
    
    Please provide:
    1. An assessment of the distribution of components by type and domain
    2. Identification of any gaps or redundancies in our component ecosystem
    3. Suggestions for organization improvements
    4. Recommendations for potential new components we should develop
    5. Advice on evolution strategies for our existing components
    
    Provide a comprehensive analysis with actionable recommendations.
    """
    
    print("\nAnalyzing library health...")
    analysis_result = await system_agent.run(analysis_prompt)
    
    print("\nLibrary health analysis:")
    print(analysis_result.result.text)

async def setup_demo_library():
    """Set up a library for the demonstration"""
    library_path = "beeai_evolution_demo.json"
    
    # Delete existing file if it exists
    if os.path.exists(library_path):
        os.remove(library_path)
        print(f"Deleted existing library at {library_path}")
    
    # Create a new library
    library = SmartLibrary(library_path)
    print(f"Created new library at {library_path}")
    
    return library_path

async def main():
    try:
        print("\n" + "="*80)
        print("BEEAI AGENT EVOLUTION DEMONSTRATION")
        print("="*80)
        
        # Ensure BeeAI templates exist
        ensure_templates_exist()
        
        # Initialize components with dependency container
        container = DependencyContainer()
        
        # Set up library
        library_path = await setup_demo_library()
        
        # Initialize services and components
        llm_service = LLMService(provider="openai", model="gpt-4o")
        container.register('llm_service', llm_service)
        
        smart_library = SmartLibrary(library_path, container=container)
        container.register('smart_library', smart_library)
        
        firmware = Firmware()
        container.register('firmware', firmware)
        
        agent_bus = SmartAgentBus(
            storage_path="smart_agent_bus.json", 
            log_path="agent_bus_logs.json",
            container=container
        )
        container.register('agent_bus', agent_bus)
        
        # Create provider registry
        provider_registry = ProviderRegistry()
        provider_registry.register_provider(BeeAIProvider(llm_service))
        container.register('provider_registry', provider_registry)
        
        # Initialize system agent
        print("\nInitializing System Agent...")
        system_agent = await SystemAgentFactory.create_agent(container=container)
        container.register('system_agent', system_agent)
        
        # Initialize components
        await smart_library.initialize()
        await agent_bus.initialize_from_library()
        
        # Create evolution tracker
        evolution_tracker = AgentEvolutionTracker()
        
        # PHASE 1: Component Creation
        sentiment_tool_id, _ = await create_sentiment_analysis_tool(system_agent)
        
        # Let's wait briefly to ensure the component is properly registered
        await asyncio.sleep(1)
        
        agent_id, _ = await create_customer_feedback_agent(system_agent, sentiment_tool_id)
        
        # Let's wait again to ensure the component is properly registered
        await asyncio.sleep(1)
        
        # Store original IDs for comparison
        component_ids = {
            "original_sentiment_tool": sentiment_tool_id,
            "original_feedback_agent": agent_id,
        }
        
        # PHASE 2: Standard Evolution
        evolved_tool_id, evolve_result = await evolve_sentiment_tool(system_agent, sentiment_tool_id)
        component_ids["evolved_sentiment_tool"] = evolved_tool_id
        
        # Record the evolution
        if sentiment_tool_id and evolved_tool_id:
            evolution_tracker.record_evolution(
                sentiment_tool_id,
                evolved_tool_id,
                "standard",
                "Added multi-language support and emotion detection"
            )
        
        # Let's wait briefly to ensure the component is properly registered
        await asyncio.sleep(1)
        
        # PHASE 3: Domain Adaptation
        financial_agent_id, adapt_result = await adapt_to_finance_domain(system_agent, agent_id)
        component_ids["financial_agent"] = financial_agent_id
        
        # Record the adaptation
        if agent_id and financial_agent_id:
            evolution_tracker.record_evolution(
                agent_id,
                financial_agent_id,
                "domain_adaptation",
                "Adapted from customer service to finance domain"
            )
        
        # Let's wait briefly to ensure the component is properly registered
        await asyncio.sleep(1)
        
        # PHASE 4: Aggressive Evolution
        web_analyzer_id, transform_result = await transform_to_web_analyzer(system_agent, evolved_tool_id)
        component_ids["web_analyzer"] = web_analyzer_id
        
        # Record the transformation
        if evolved_tool_id and web_analyzer_id:
            evolution_tracker.record_evolution(
                evolved_tool_id,
                web_analyzer_id,
                "aggressive",
                "Transformed from sentiment analyzer to web content analyzer"
            )
        
        # Let's wait briefly to ensure the component is properly registered
        await asyncio.sleep(1)
        
        # PHASE 5: Testing Evolved Components
        await test_evolved_components(
            system_agent,
            smart_library,
            {
                "sentiment_tool": evolved_tool_id,
                "financial_agent": financial_agent_id,
                "web_analyzer": web_analyzer_id
            }
        )
        
        # PHASE 6: Component Comparison
        await compare_components(
            system_agent, 
            smart_library,
            component_ids.get("original_sentiment_tool"),
            component_ids.get("evolved_sentiment_tool")
        )
        
        # PHASE 7: Library Health Analysis
        await analyze_library_health(system_agent, smart_library)
        
        print("\n" + "="*80)
        print("BEEAI AGENT EVOLUTION DEMONSTRATION COMPLETED")
        print("="*80)
        
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())