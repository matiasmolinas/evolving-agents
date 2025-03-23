import asyncio
import json
import os
from datetime import datetime
from typing import List, Dict, Any

from evolving_agents.agents.architect_zero import create_architect_zero
from evolving_agents.smart_library.smart_library import SmartLibrary
from evolving_agents.core.system_agent import SystemAgentFactory
from evolving_agents.core.llm_service import LLMService
from evolving_agents.agent_bus.smart_agent_bus import SmartAgentBus

# Types for our memory system
class UserMessage:
    def __init__(self, text: str, timestamp: datetime = None):
        self.text = text
        self.timestamp = timestamp or datetime.now()
        self.importance = 0.5  # Default importance

class Memory:
    def __init__(self, content: str, source: str, timestamp: datetime = None, importance: float = 0.5, metadata: Dict = None):
        self.content = content
        self.source = source
        self.timestamp = timestamp or datetime.now()
        self.importance = importance
        self.metadata = metadata or {}

async def run_personal_ai(
    prompt: str, 
    user_messages_path: str, 
    existing_memories_path: str = None, 
    output_path: str = None
) -> Dict[str, Any]:
    """
    Run the personal AI with previous messages and memories.
    
    Args:
        prompt: Architect prompt describing what to build
        user_messages_path: Path to JSON file with recent user messages
        existing_memories_path: Optional path to JSON file with existing memories
        output_path: Optional path to save the results
        
    Returns:
        Dictionary with the AI's response and updated memories
    """
    # Step 1: Load recent user messages
    with open(user_messages_path, "r") as f:
        user_messages_data = json.load(f)
        user_messages = [UserMessage(msg["text"], datetime.fromisoformat(msg["timestamp"])) 
                        for msg in user_messages_data]
    
    # Step 2: Load existing memories if available
    existing_memories = []
    if existing_memories_path and os.path.exists(existing_memories_path):
        with open(existing_memories_path, "r") as f:
            memories_data = json.load(f)
            existing_memories = [
                Memory(
                    content=mem["content"],
                    source=mem["source"],
                    timestamp=datetime.fromisoformat(mem["timestamp"]),
                    importance=mem["importance"],
                    metadata=mem.get("metadata", {})
                ) 
                for mem in memories_data
            ]
    
    # Step 3: Set up services
    llm_service = LLMService(provider="openai", model="gpt-4o")
    smart_library = SmartLibrary("personal_ai_library.json")
    
    # Create the system agent
    system_agent = await SystemAgentFactory.create_agent(
        llm_service=llm_service,
        smart_library=smart_library
    )
    
    # Create and initialize the Smart Agent Bus
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
    
    # Step 5: Prepare input data for architect
    # Format recent messages for the architect
    formatted_messages = "\n\n".join([
        f"USER ({msg.timestamp.strftime('%Y-%m-%d %H:%M')}): {msg.text}" 
        for msg in user_messages
    ])
    
    # Format existing memories
    formatted_memories = "\n\n".join([
        f"MEMORY ({mem.timestamp.strftime('%Y-%m-%d %H:%M')}) [{mem.source}] (Importance: {mem.importance:.1f}): {mem.content}"
        for mem in existing_memories
    ])
    
    # Step 6: Run the full task by passing prompt and data
    full_prompt = (
        f"{prompt}\n\n"
        f"CURRENT USER MESSAGES:\n{formatted_messages}\n\n"
        f"EXISTING MEMORIES:\n{formatted_memories}\n\n"
        f"IMPORTANT: Your output MUST include these three sections clearly separated with headers:\n"
        f"1. 'AI RESPONSE:' - The direct response to give to the user\n"
        f"2. 'UPDATED MEMORIES:' - List of memories to store (including both new and consolidated)\n"
        f"3. 'AGENTS USED:' - List of the specialized agents that were used in this process\n\n"
        f"For each memory in UPDATED MEMORIES, include Content, Source, Importance, and Metadata.\n"
    )
    
    # Run the architect to design and implement the system
    result = await architect.run(full_prompt)
    
    # Step 7: Extract the result text
    if hasattr(result, 'result') and hasattr(result.result, 'text'):
        result_text = result.result.text
    else:
        result_text = str(result)
    
    # Step 8: Create direct LLM parsing of the result for more reliable extraction
    parsing_prompt = f"""
    The following is output from an AI system that should contain three sections: 
    AI RESPONSE, UPDATED MEMORIES, and AGENTS USED.
    
    Please extract and format these sections into a valid JSON object with three keys:
    - ai_response: The text response to the user
    - agents_used: Array of agent descriptions
    - memories: Array of memory objects with content, source, timestamp, importance, and metadata
    
    If any section is missing, use an empty string, empty array, or empty object as appropriate.
    Current timestamp for new memories: {datetime.now().isoformat()}
    
    OUTPUT TO PARSE:
    {result_text}
    
    RETURN ONLY VALID JSON:
    """
    
    # Parse the output using the LLM
    parsed_json = await llm_service.generate(parsing_prompt)
    
    # Try to load the JSON, with fallback mechanisms
    try:
        output_data = json.loads(parsed_json)
    except json.JSONDecodeError:
        # Fallback: Try to extract just the JSON portion
        json_start = parsed_json.find('{')
        json_end = parsed_json.rfind('}') + 1
        if json_start >= 0 and json_end > json_start:
            try:
                output_data = json.loads(parsed_json[json_start:json_end])
            except:
                # Second fallback: Create basic structure
                output_data = {
                    "ai_response": "I'm sorry, I couldn't process your messages properly. Could you try again?",
                    "agents_used": ["ErrorRecoveryAgent"],
                    "memories": [
                        {
                            "content": "System encountered error processing user input",
                            "source": "system_error",
                            "timestamp": datetime.now().isoformat(),
                            "importance": 0.3,
                            "metadata": {"error": "JSON parsing failed"}
                        }
                    ]
                }
    
    # Ensure all required fields exist
    if "ai_response" not in output_data:
        output_data["ai_response"] = ""
    if "agents_used" not in output_data:
        output_data["agents_used"] = []
    if "memories" not in output_data:
        output_data["memories"] = []
    
    # Step 9: Save the result if output path provided
    if output_path:
        with open(output_path, "w") as f:
            json.dump(output_data, f, indent=2)
        print(f"Output saved to {output_path}")
    
    # Convert memories to objects
    memory_objects = []
    for mem in output_data["memories"]:
        # Ensure timestamp exists
        if "timestamp" not in mem:
            mem["timestamp"] = datetime.now().isoformat()
        
        memory_objects.append(Memory(
            content=mem["content"],
            source=mem["source"],
            timestamp=datetime.fromisoformat(mem["timestamp"]) if isinstance(mem["timestamp"], str) else mem["timestamp"],
            importance=mem.get("importance", 0.5),
            metadata=mem.get("metadata", {})
        ))
    
    # Step 10: Return structured result
    return {
        "ai_response": output_data["ai_response"],
        "agents_used": output_data["agents_used"],
        "memories": memory_objects
    }

# ---- MAIN ----
if __name__ == "__main__":
    ARCHITECT_PROMPT = """
    Build a single-threaded personal AI assistant that maintains memory and context over time.
    
    REQUIREMENTS:
    1. Create a system that can respond to the user based on immediate context AND long-term memories
    2. Implement multi-tier memory compression:
       - HOT: Recent and frequently accessed memories (high detail)
       - WARM: Important but less recent memories (moderate detail)
       - COLD: Old memories with low importance (compressed summaries)
    3. Design memory importance ranking based on:
       - Recency: When was this last relevant?
       - Frequency: How often does this come up?
       - Emotional significance: How important is this to the user?
       - Uniqueness: Is this information rare or common?
    4. Implement memory consolidation agents that compress related memories periodically
    5. Add detail retrieval agents that can expand compressed memories when needed
    
    Your task is to:
    1. Process the user's recent messages
    2. Retrieve and prioritize relevant memories
    3. Consolidate related memories when appropriate
    4. Generate a natural, contextually aware response
    5. Update the memory store with new information
    
    EXAMPLE SCENARIO:
    The user is discussing dinner plans, work, and family. You need to:
    - Check if the user has relevant food preferences or restrictions
    - Note the work accomplishment in memory
    - Connect the mother's birthday information with existing knowledge
    - Respond naturally while showing memory of past conversations
    
    EXPECTED OUTPUT FORMAT:
    AI RESPONSE:
    [The response to send to the user]
    
    UPDATED MEMORIES:
    - Content: [Memory content]
      Source: [Where this memory came from]
      Importance: [0.0-1.0 score]
      Metadata: [Additional structured data]
    
    AGENTS USED:
    - [List of specialized agents utilized]
    """
    
    # Run the personal AI
    result = asyncio.run(run_personal_ai(
        prompt=ARCHITECT_PROMPT,
        user_messages_path="user_messages.json",
        existing_memories_path="existing_memories.json",
        output_path="personal_ai_output.json"
    ))
    
    # Print the response
    print("\n=== AI RESPONSE ===")
    print(result["ai_response"])
    
    print("\n=== AGENTS USED ===")
    for agent in result["agents_used"]:
        print(f"- {agent}")
    
    print(f"\n=== {len(result['memories'])} MEMORIES UPDATED ===")
    for i, memory in enumerate(result["memories"], 1):
        print(f"{i}. {memory.content} (Importance: {memory.importance:.1f})")
    
    print("\nFull output saved to personal_ai_output.json")