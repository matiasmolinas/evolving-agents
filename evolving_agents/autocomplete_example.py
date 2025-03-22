# evolving_agents/autocomplete_example.py
import asyncio
import os
import sys
import json
from run_architect import run_architect_agent, initialize_system, print_step

# Sample user interaction sequences across different applications
SAMPLE_DATA = {
    "twitter": [
        "Just read a fascinating article about large language models like GPT-4 and how they're transforming software development.",
        "OpenAI's latest DALL-E 3 model generates incredibly photorealistic images from text prompts. The quality is mindblowing.",
        "Apple's new M3 chip benchmarks show significant performance improvements over the M2, especially for machine learning tasks."
    ],
    "slack": [
        "Hey team, can we discuss the LLM integration project in our next meeting?",
        "I'm trying to optimize our prompt engineering approach for the new application.",
        "Has anyone tested how DALL-E 3 handles technical diagrams compared to Midjourney?"
    ],
    "email": [
        "Subject: Project Timeline Update\nHi everyone,\nAttached is the updated timeline for the AI integration project. We need to finalize the LLM provider selection by Friday.",
        "Subject: Meeting Notes: ML Infrastructure\nTeam,\nDuring yesterday's meeting we decided to proceed with containerized deployment for all ML models including the new GPT-based services.",
        "Subject: Vendor Comparison\nThe attached spreadsheet compares OpenAI, Anthropic, and Cohere pricing models for our expected usage patterns."
    ]
}

# Sample autocomplete input scenarios
AUTOCOMPLETE_SCENARIOS = [
    {"partial_text": "Let me check the latest ", "current_app": "slack"},
    {"partial_text": "We should implement the GPT", "current_app": "email"},
    {"partial_text": "The ML infra", "current_app": "slack"},
    {"partial_text": "DALL-E can", "current_app": "twitter"},
    {"partial_text": "The benchmarks for the M", "current_app": "email"}
]

async def main():
    print_step("SMART AUTOCOMPLETE SYSTEM EXAMPLE", 
             "This demonstration shows how a context-aware autocomplete system maintains context across applications", 
             "INFO")
    
    # Create data directory if it doesn't exist
    os.makedirs("data", exist_ok=True)
    
    # Check if the library file exists
    if not os.path.exists("data/autocomplete_library.json"):
        print_step("MISSING LIBRARY FILE", 
                 "Please run generate_autocomplete_library.py first to create the required library file", 
                 "ERROR")
        sys.exit(1)
    
    # Initialize the system with the autocomplete library
    system = await initialize_system("data/autocomplete_library.json")
    
    # Create prompts directory if it doesn't exist
    os.makedirs("prompts", exist_ok=True)
    
    # Check if the prompt file exists
    if not os.path.exists("prompts/autocomplete_prompt.txt"):
        print_step("MISSING PROMPT FILE", 
                 "Please create the autocomplete prompt file at prompts/autocomplete_prompt.txt", 
                 "ERROR")
        sys.exit(1)
    
    # Print input scenario data for visibility
    print_step("INPUT SCENARIO DATA", "The system will process these interactions across different applications", "INFO")
    for app, messages in SAMPLE_DATA.items():
        print(f"\n{app.upper()} MESSAGES:")
        for i, message in enumerate(messages, 1):
            print(f"{i}. {message[:100]}{'...' if len(message) > 100 else ''}")
    
    print("\nAUTOCOMPLETE TEST SCENARIOS:")
    for i, scenario in enumerate(AUTOCOMPLETE_SCENARIOS, 1):
        print(f"{i}. App: {scenario['current_app']}, Input: \"{scenario['partial_text']}\"")
    
    # Run the architect agent with the autocomplete prompt
    result = await run_architect_agent(
        system,
        "prompts/autocomplete_prompt.txt",
        json.dumps({"sample_data": SAMPLE_DATA, "autocomplete_scenarios": AUTOCOMPLETE_SCENARIOS}),
        "autocomplete"
    )
    
    # Print execution results summary
    if hasattr(result, 'result') and hasattr(result.result, 'text'):
        # Try to extract the autocomplete suggestions from the agent's output
        import re
        suggestions_pattern = r'AUTOCOMPLETE SUGGESTIONS:(.*?)(?:ANALYSIS:|$)'
        match = re.search(suggestions_pattern, result.result.text, re.DOTALL)
        
        if match:
            print_step("AUTOCOMPLETE SUGGESTIONS SUMMARY", match.group(1).strip(), "SUCCESS")
        else:
            # Look for any JSON blocks containing suggestions
            json_pattern = r'```json\s*(.*?)\s*```'
            json_matches = re.findall(json_pattern, result.result.text, re.DOTALL)
            
            if json_matches:
                for i, json_str in enumerate(json_matches, 1):
                    try:
                        data = json.loads(json_str)
                        if isinstance(data, list) and len(data) > 0 and "completion" in data[0]:
                            print_step(f"AUTOCOMPLETE SUGGESTIONS SET {i}", json_str, "SUCCESS")
                    except:
                        pass
    
    # Analyze the results with LLM
    analysis_prompt = """
    Analyze how well the system achieved the goal of maintaining context across applications
    and providing smart autocomplete suggestions. Consider:
    
    1. Did the system successfully extract and maintain context across different applications?
    2. Were the autocomplete suggestions contextually relevant?
    3. Did the system demonstrate understanding of technical terms from previous contexts?
    4. How did the system adapt to different application contexts?
    5. What are the strengths and limitations of this approach?
    
    Provide a concise analysis of the system's performance.
    """
    
    llm_service = system["llm_service"]
    analysis_result = await llm_service.generate(analysis_prompt)
    
    print_step("PERFORMANCE ANALYSIS", analysis_result, "INFO")
    
    print_step("AUTOCOMPLETE EXAMPLE COMPLETED", 
             "The smart autocomplete system demonstration has been executed", 
             "INFO")

if __name__ == "__main__":
    asyncio.run(main())