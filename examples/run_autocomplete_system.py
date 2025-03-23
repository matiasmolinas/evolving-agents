# examples/run_autocomplete_system.py

import asyncio
import json
from typing import List
from evolving_agents.agents.architect_zero import create_architect_zero
from evolving_agents.smart_library.smart_library import SmartLibrary
from evolving_agents.core.system_agent import SystemAgentFactory
from evolving_agents.core.llm_service import LLMService
from evolving_agents.agent_bus.smart_agent_bus import SmartAgentBus


async def run_smart_autocomplete(
    prompt: str, 
    inputs: List[dict], 
    output_path: str
):
    """
    Run a context-aware autocomplete system that learns from multiple sources.
    
    Args:
        prompt: Instructions for the Architect agent
        inputs: List of dictionaries containing texts from different sources
        output_path: Where to save the autocomplete suggestions
    """
    # Step 1: Set up core services
    llm_service = LLMService(provider="openai", model="gpt-4o")
    smart_library = SmartLibrary("smart_autocomplete_library.json")
    
    # Step 2: Create the system agent
    system_agent = await SystemAgentFactory.create_agent(
        llm_service=llm_service,
        smart_library=smart_library
    )

    # Step 3: Create and initialize the Smart Agent Bus
    agent_bus = SmartAgentBus(
        smart_library=smart_library,
        system_agent=system_agent,
        storage_path="smart_agent_bus.json",
        log_path="agent_bus_logs.json"
    )
    await agent_bus.initialize_from_library()

    # Step 4: Create the architect agent
    architect = await create_architect_zero(
        llm_service=llm_service,
        smart_library=smart_library,
        agent_bus=agent_bus,
        system_agent_factory=SystemAgentFactory.create_agent
    )

    # Step 5: Format inputs into a single context object
    formatted_inputs = json.dumps(inputs, indent=2)
    
    # Step 6: Run the full task by combining prompt and inputs
    full_prompt = (
        f"{prompt}\n\n"
        f"Here are the input contexts from various sources:\n\n"
        f"{formatted_inputs}"
    )

    result = await architect.run(full_prompt)

    # Step 7: Extract and save the result
    if hasattr(result, 'result') and hasattr(result.result, 'text'):
        result_text = result.result.text
    elif hasattr(result, 'text'):
        result_text = result.text
    else:
        result_text = str(result)

    with open(output_path, "w") as f:
        f.write(result_text)

    print(f"Smart autocomplete system completed. Suggestions saved to {output_path}")


if __name__ == "__main__":
    # Sample contexts from different sources
    contexts = [
        {
            "source": "Twitter/X",
            "timestamp": "2023-10-15T09:30:00",
            "text": "Just read about Project Nightingale at Google. Their approach to quantum computing is revolutionary. #QuantumAI #ProjectNightingale"
        },
        {
            "source": "Email",
            "timestamp": "2023-10-15T10:15:00",
            "text": "Hi team, Regarding our upcoming presentation on machine learning applications in healthcare, please review the slides I've attached. We should emphasize our competitive advantage over Acme Health's recent predictive analytics system."
        },
        {
            "source": "Slack",
            "timestamp": "2023-10-15T11:45:00",
            "text": "The meeting with Dr. Zhang went well! She's interested in our proposal for implementing federated learning across multiple hospitals while maintaining HIPAA compliance. We should follow up next week."
        },
        {
            "source": "Google Doc",
            "timestamp": "2023-10-15T14:20:00",
            "text": "Draft: Healthcare AI Strategy 2024\n\nObjectives:\n1. Deploy predictive analytics for patient readmission\n2. Implement federated learning across partner hospitals\n3. Obtain FDA approval for our diagnostic algorithm\n4. Compete effectively against Acme Health's platform"
        },
        {
            "source": "Text Message",
            "timestamp": "2023-10-15T16:05:00",
            "text": "Don't forget we need to prepare for the Nightingale demo tomorrow! Make sure the quantum simulation is working properly."
        }
    ]

    # Architect prompt
    architect_prompt = """
    Design a context-aware, adaptive autocomplete system that:

    1. Analyzes multiple text sources to build a unified user context profile
    2. Identifies key entities, terms, projects, and relationships important to the user
    3. Generates intelligent autocomplete suggestions that incorporate:
       - Domain-specific terminology and jargon
       - Project codenames and internal references
       - People, places, and organizations from the user's context
       - Recent topics the user has been engaging with
    4. Adapts suggestions based on which application the user is currently using
    5. Maintains privacy by processing all data locally
    
    The system should synthesize information across contexts to provide suggestions 
    that demonstrate true understanding rather than simple pattern matching.
    
    For example, if a user has been discussing "Project Nightingale" on Twitter 
    and later types "We need to prepare for the Night..." in a work email, 
    the system should suggest "Nightingale" and understand the context around it.
    
    Your task is to design this system and demonstrate how it would generate 
    intelligent autocomplete suggestions for a few example scenarios.
    """

    # Run the autocomplete system
    asyncio.run(run_smart_autocomplete(
        prompt=architect_prompt,
        inputs=contexts,
        output_path="autocomplete_suggestions.txt"
    ))