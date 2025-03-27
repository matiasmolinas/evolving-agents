# examples/forms/run_conversational_form.py

import asyncio
import json
import re
import os
from evolving_agents.agents.architect_zero import create_architect_zero
from evolving_agents.smart_library.smart_library import SmartLibrary
from evolving_agents.core.system_agent import SystemAgentFactory
from evolving_agents.core.llm_service import LLMService
from evolving_agents.agent_bus.smart_agent_bus import SmartAgentBus
from evolving_agents.core.dependency_container import DependencyContainer

import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

async def create_conversational_form(form_prompt: str, form_id: str = "feedback_form"):
    """Create a conversational form based on natural language description."""
    
    # Create dependency container to manage component dependencies
    container = DependencyContainer()
    
    # Initialize services
    llm_service = LLMService(provider="openai", model="gpt-4o")
    container.register('llm_service', llm_service)
    
    # Make sure library directory exists
    if not os.path.exists("form_library.json"):
        with open("form_library.json", "w") as f:
            f.write("[]")
    
    smart_library = SmartLibrary("form_library.json", container=container)
    container.register('smart_library', smart_library)
    
    # Create agent bus with null system agent
    agent_bus = SmartAgentBus(
        storage_path="form_agent_bus.json",
        log_path="form_agent_logs.json",
        container=container
    )
    container.register('agent_bus', agent_bus)
    
    # Create the system agent
    system_agent = await SystemAgentFactory.create_agent(container=container)
    container.register('system_agent', system_agent)
    
    # Initialize components
    await smart_library.initialize()
    await agent_bus.initialize_from_library()
    
    # Create the architect agent using the container
    architect = await create_architect_zero(container=container)
    
    # Run the architect to create the form system
    form_creation_prompt = f"""
    Design a conversational form system based on this form description:
    
    FORM DESCRIPTION:
    {form_prompt}
    
    Create a complete system with these NAMED COMPONENTS:
    1. FormDefinitionParser - for extracting questions and logic from the form definition
    2. ConversationFlowManager - for managing the conversation flow and branching logic
    3. ResponseValidator - for validating user responses based on question type
    4. ResponseProcessor - for processing and storing validated responses
    5. FormSummaryGenerator - for creating summaries of all collected responses
    
    For each component:
    - Give it a clear name
    - Define its purpose and responsibilities 
    - Explain how it interacts with other components
    - Provide sample code showing its implementation
    
    Generate a complete implementation with all necessary components.
    """
    
    result = await architect.run(form_creation_prompt)
    
    # Extract agent names from the result using regex for better detection
    agents_used = []
    # Look for agent/component names in various patterns
    agent_patterns = [
        r"class\s+(\w+)(?:Agent|Manager|Processor|Parser|Generator|Validator)",
        r"def\s+create_(\w+)(?:Agent|Manager|Processor|Parser|Generator|Validator)",
        r"['\"]*name['\"]*\s*:\s*['\"]*(\w+)(?:Agent|Manager|Processor|Parser|Generator|Validator)['\"]"
    ]
    
    for pattern in agent_patterns:
        for match in re.finditer(pattern, result.result.text, re.IGNORECASE):
            agent_name = match.group(1)
            # Add suffixes if they're missing but implied
            if not any(suffix in agent_name for suffix in ["Agent", "Manager", "Processor", "Parser", "Generator", "Validator"]):
                # Add appropriate suffix based on context
                if "valid" in result.result.text[max(0, match.start()-20):min(len(result.result.text), match.end()+20)].lower():
                    agent_name += "Validator"
                elif "process" in result.result.text[max(0, match.start()-20):min(len(result.result.text), match.end()+20)].lower():
                    agent_name += "Processor"
                elif "parse" in result.result.text[max(0, match.start()-20):min(len(result.result.text), match.end()+20)].lower():
                    agent_name += "Parser"
                elif "generat" in result.result.text[max(0, match.start()-20):min(len(result.result.text), match.end()+20)].lower():
                    agent_name += "Generator"
                elif "manag" in result.result.text[max(0, match.start()-20):min(len(result.result.text), match.end()+20)].lower():
                    agent_name += "Manager"
                else:
                    agent_name += "Agent"
            
            # Clean up the name and add if not already in list
            agent_name = agent_name.strip()
            if agent_name and agent_name not in agents_used and len(agent_name) > 3:
                agents_used.append(agent_name)
    
    # If no agents are found, extract any capitalized names that might be components
    if not agents_used:
        # Look for capitalized words that might be component names
        for match in re.finditer(r"\b([A-Z][a-zA-Z]+(?:Agent|Manager|Processor|Parser|Generator|Validator|Tool))\b", result.result.text):
            agent_name = match.group(1)
            if agent_name not in agents_used:
                agents_used.append(agent_name)
    
    # If still no agents found, use the default names from the prompt
    if not agents_used:
        agents_used = [
            "FormDefinitionParser", 
            "ConversationFlowManager", 
            "ResponseValidator", 
            "ResponseProcessor", 
            "FormSummaryGenerator"
        ]
    
    # Save the form definition
    with open(f"{form_id}_definition.json", "w") as f:
        f.write(json.dumps({
            "form_id": form_id,
            "form_prompt": form_prompt,
            "agents_used": agents_used
        }, indent=2))
    
    print(f"Form system design complete. Agents identified: {len(agents_used)}")
    
    return {
        "status": "success",
        "form_id": form_id,
        "agents_used": agents_used,
        "message": f"Conversational form created with ID: {form_id}",
        "full_response": result.result.text
    }

async def fill_out_form(form_id: str, user_responses: list):
    """Process a user filling out the conversational form."""
    
    # Create dependency container to manage component dependencies
    container = DependencyContainer()
    
    # Initialize services
    llm_service = LLMService(provider="openai", model="gpt-4o")
    container.register('llm_service', llm_service)
    
    smart_library = SmartLibrary("form_library.json", container=container)
    container.register('smart_library', smart_library)
    
    # Create agent bus with null system agent
    agent_bus = SmartAgentBus(
        storage_path="form_agent_bus.json",
        log_path="form_agent_logs.json",
        container=container
    )
    container.register('agent_bus', agent_bus)
    
    # Create the system agent
    system_agent = await SystemAgentFactory.create_agent(container=container)
    container.register('system_agent', system_agent)
    
    # Initialize components
    await smart_library.initialize()
    await agent_bus.initialize_from_library()
    
    # Load the form definition
    try:
        with open(f"{form_id}_definition.json", "r") as f:
            form_definition = json.load(f)
    except FileNotFoundError:
        return {
            "status": "error",
            "message": f"Form with ID {form_id} not found"
        }
    
    # Process each user response through the form system
    all_results = []
    for i, response in enumerate(user_responses):
        # Create prompt for handling this user response
        process_prompt = f"""
        Process this user response for form ID: {form_id}
        
        FORM DESCRIPTION:
        {form_definition['form_prompt']}
        
        USER RESPONSE: {response}
        RESPONSE INDEX: {i+1} of {len(user_responses)}
        
        Process this response and determine:
        1. Which question this is answering
        2. Whether the response is valid
        3. What the next question should be (if any)
        4. Store the validated response
        
        Return the processed response and the next question to ask.
        """
        
        result = await system_agent.run(process_prompt)
        all_results.append({
            "user_response": response,
            "processed_result": result.result.text
        })
    
    # Generate final summary of all responses
    summary_prompt = f"""
    Generate a summary of all responses for form ID: {form_id}
    
    FORM DESCRIPTION:
    {form_definition['form_prompt']}
    
    PROCESSED RESPONSES:
    {json.dumps([{"response": r["user_response"]} for r in all_results], indent=2)}
    
    Create a structured summary of all responses in a clean, organized format.
    Include analytics about the responses if relevant.
    """
    
    summary_result = await system_agent.run(summary_prompt)
    
    # Save the responses
    with open(f"{form_id}_responses.json", "w") as f:
        f.write(json.dumps({
            "form_id": form_id,
            "responses": all_results,
            "summary": summary_result.result.text
        }, indent=2))
    
    return {
        "status": "success",
        "form_id": form_id,
        "response_count": len(user_responses),
        "summary": summary_result.result.text
    }

async def main():
    # Example 1: Customer Feedback Form
    feedback_form_prompt = """
    Create a customer feedback form for our restaurant with these questions:
    1. How would you rate your dining experience from
    1-10?
    2. What did you order? (food items)
    3. Was the service prompt and friendly?
    4. What could we improve?
    5. Would you recommend us to friends and family?
    
    If they rate below 5 on question 1, ask what specifically went wrong.
    If they mention "slow" in any answer, ask about wait times.
    Thank them at the end and offer a 10% discount code for their next visit.
    """
    
    print("\n=== Creating Restaurant Feedback Form ===")
    form_result = await create_conversational_form(
        form_prompt=feedback_form_prompt,
        form_id="restaurant_feedback"
    )
    
    print("Form Created!")
    print(f"Agents used: {form_result['agents_used']}")
    
    # Sample user responses to the form
    user_responses = [
        "I'd give it a 7 out of 10",
        "I had the pasta carbonara and a glass of red wine",
        "Yes, our server was really friendly but the food was a bit slow to arrive",
        "The pasta was slightly overcooked. Maybe have more staff during rush hour?",
        "Yes, I'd recommend it to friends who aren't in a hurry"
    ]
    
    print("\n=== Processing Restaurant Feedback Responses ===")
    # Process the responses
    form_responses = await fill_out_form(
        form_id="restaurant_feedback",
        user_responses=user_responses
    )
    
    print("\nForm Responses Processed!")
    print("Summary:")
    print(form_responses["summary"])
    
    # Example 2: Job Application Form
    job_form_prompt = """
    Create a job application form for a software developer position with these requirements:
    1. Ask for their name and contact information
    2. Ask about their years of experience with Python, JavaScript, and cloud platforms
    3. Ask them to describe their most challenging project
    4. Ask why they want to work with our company
    5. Ask about their salary expectations
    6. Ask when they would be available to start
    
    If they have less than 2 years of experience, ask about their educational background.
    If they mention "machine learning" or "AI", ask about specific ML projects.
    Thank them for their application and explain the next steps in our process.
    """
    
    print("\n=== Creating Job Application Form ===")
    job_form_result = await create_conversational_form(
        form_prompt=job_form_prompt,
        form_id="job_application"
    )
    
    print("Form Created!")
    print(f"Agents used: {job_form_result['agents_used']}")
    
    # Process another set of responses for the job application
    candidate1_responses = [
        "My name is Alex Johnson, email alex@example.com, phone 555-123-4567",
        "I have 4 years of Python experience, 2 years of JavaScript, and 3 years working with AWS",
        "My most challenging project was building a machine learning system for predictive maintenance in manufacturing",
        "I'm interested in your company's work on sustainable technology and the collaborative culture I've heard about",
        "I'm looking for a salary in the range of $90-110k depending on benefits",
        "I could start in two weeks after giving notice to my current employer"
    ]
    
    print("\n=== Processing Job Application Responses ===")
    job_responses = await fill_out_form(
        form_id="job_application",
        user_responses=candidate1_responses
    )
    
    print("\nJob Application Processed!")
    print("Summary:")
    print(job_responses["summary"])
    
    # Example 3: Event Registration Form
    event_form_prompt = """
    Create a registration form for our tech conference with these questions:
    1. What is your name and email address?
    2. Which ticket type do you want? (Options: General, VIP, Student)
    3. Which workshops do you want to attend? (Can select multiple from: Web Development, Mobile Apps, Machine Learning, Cloud Computing, Cybersecurity)
    4. Do you have any dietary restrictions?
    5. How did you hear about our event?
    
    If they select Student, ask for their school name and ID.
    If they select more than 2 workshops, confirm they understand there may be scheduling conflicts.
    Thank them for registering and tell them confirmation details will be emailed.
    """
    
    print("\n=== Creating Event Registration Form ===")
    event_form_result = await create_conversational_form(
        form_prompt=event_form_prompt,
        form_id="event_registration"
    )
    
    print("Form Created!")
    print(f"Agents used: {event_form_result['agents_used']}")
    
    # Event registration responses
    event_responses = [
        "Jane Smith, jane.smith@example.com",
        "I'd like a Student ticket please",
        "University of Technology, student ID: UT2023456",
        "I'm interested in the Web Development, Machine Learning, and Cloud Computing workshops",
        "Yes, I understand there might be scheduling conflicts with three workshops, but I'd like to keep my options open",
        "I'm vegetarian", 
        "I found out about this from a LinkedIn post"
    ]
    
    print("\n=== Processing Event Registration Responses ===")
    event_result = await fill_out_form(
        form_id="event_registration",
        user_responses=event_responses
    )
    
    print("\nEvent Registration Processed!")
    print("Summary:")
    print(event_result["summary"])
    
    print("\n=== All forms processed successfully! ===")
    print("Files created:")
    print(f"  - restaurant_feedback_definition.json")
    print(f"  - restaurant_feedback_responses.json")
    print(f"  - job_application_definition.json")
    print(f"  - job_application_responses.json")
    print(f"  - event_registration_definition.json")
    print(f"  - event_registration_responses.json")

if __name__ == "__main__":
    asyncio.run(main())