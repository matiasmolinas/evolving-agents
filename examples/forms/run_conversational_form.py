# examples/forms/run_conversational_form.py

import asyncio
import json
import re
import os
import logging
from evolving_agents.agents.architect_zero import create_architect_zero
from evolving_agents.smart_library.smart_library import SmartLibrary
from evolving_agents.core.system_agent import SystemAgentFactory
from evolving_agents.core.llm_service import LLMService
from evolving_agents.agent_bus.smart_agent_bus import SmartAgentBus
from evolving_agents.core.dependency_container import DependencyContainer
from evolving_agents.core.mongodb_client import MongoDBClient
from datetime import datetime, timezone
from evolving_agents.tools.memory.experience_recorder_tool import ExperienceRecorderTool, ExperienceDataInput

import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Placeholder for Memory Manager Agent ID (to be updated with actual ID later)
MEMORY_MANAGER_AGENT_ID = "memory_manager_agent_v1"

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
    
    smart_library = SmartLibrary(container=container)
    container.register('smart_library', smart_library)
    
    # Create agent bus with null system agent
    agent_bus = SmartAgentBus(container=container)
    container.register('agent_bus', agent_bus)

    # Instantiate ExperienceRecorderTool
    experience_recorder_tool = ExperienceRecorderTool(
        agent_bus=agent_bus,
        memory_manager_agent_id=MEMORY_MANAGER_AGENT_ID
    )
    
    # Create the system agent
    system_agent = await SystemAgentFactory.create_agent(container=container)
    container.register('system_agent', system_agent)
    
    # Initialize components
    await smart_library.initialize()
    await agent_bus.initialize_from_library()
    
    # Get MongoDB Client
    mongodb_client = container.get('mongodb_client')
    if not mongodb_client:
        mongodb_client = MongoDBClient() # Assumes MONGODB_URI and MONGODB_DATABASE_NAME are in .env
        container.register('mongodb_client', mongodb_client)
        
    form_definitions_collection_name = "eat_form_definitions"
    
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
    
    # Enhanced Component Detail Extraction
    component_record_ids = []
    component_pattern = re.compile(
        r"Component Name:\s*(.*?)\s*"
        r"Purpose and Responsibilities:\s*([\s\S]*?)\s*"
        r"Sample Code:\s*\`\`\`python\s*([\s\S]*?)\`\`\`",
        re.DOTALL
    )

    extracted_components = []
    for match in component_pattern.finditer(result.result.text):
        comp_name = match.group(1).strip()
        comp_description = match.group(2).strip()
        comp_code = match.group(3).strip()

        if not comp_name:
            print("Warning: Found a component with no name. Skipping.")
            continue

        if not comp_description:
            print(f"Warning: No description found for component {comp_name}. Using placeholder.")
            comp_description = "No description provided."

        if not comp_code:
            print(f"Warning: No code snippet found for component {comp_name}. Using placeholder.")
            comp_code = "# No code snippet extracted."

        extracted_components.append({
            "name": comp_name,
            "description": comp_description,
            "code": comp_code
        })

    # If direct parsing fails, fall back to named components from the prompt
    # and use them as placeholders.
    if not extracted_components:
        print("Warning: Could not parse component details from architect response. Using default names from prompt.")
        default_component_names = [
            "FormDefinitionParser", 
            "ConversationFlowManager", 
            "ResponseValidator", 
            "ResponseProcessor", 
            "FormSummaryGenerator"
        ]
        for name in default_component_names:
            extracted_components.append({
                "name": name,
                "description": "No description provided (default component).",
                "code": "# No code snippet extracted (default component)."
            })

    # Register Components in SmartLibrary
    for comp in extracted_components:
        try:
            record_id = await smart_library.create_record(
                name=comp["name"],
                record_type="AGENT",  # Assuming these are agent-like components
                domain="conversational_form",
                description=comp["description"],
                code_snippet=comp["code"]
            )
            component_record_ids.append(record_id)
            print(f"Component '{comp['name']}' registered in SmartLibrary with ID: {record_id}")
        except Exception as e:
            print(f"Error registering component {comp['name']} in SmartLibrary: {e}")
            # Decide if you want to append a placeholder ID or skip
            # For now, let's skip if registration fails.
            # Alternatively, you could use a special marker for failed registrations.

    # Save the form definition to MongoDB
    form_definition_data = {
        "form_id": form_id,
        "form_prompt": form_prompt,
        "component_record_ids": component_record_ids, # Updated key
        "created_at": datetime.now(timezone.utc) # Store creation time
    }
    try:
        form_definitions_collection = mongodb_client.get_collection(form_definitions_collection_name)
        await form_definitions_collection.replace_one(
            {"form_id": form_id}, 
            form_definition_data, 
            upsert=True
        )
        print(f"Form definition for {form_id} saved to MongoDB collection '{form_definitions_collection_name}'.")
    except Exception as e:
        print(f"Error saving form definition {form_id} to MongoDB: {e}")
        # Optionally, re-raise or handle as appropriate for the application
    
    print(f"Form system design complete. Components processed: {len(component_record_ids)}")

    # Prepare and record the experience
    if component_record_ids: # Only record if components were processed
        experience_input = ExperienceDataInput(
            primary_goal_description="System to create a conversational form.",
            sub_task_description=form_prompt, # The original prompt used to create the form
            initiating_agent_id="form_creation_service", # Or "architect_zero"
            final_outcome="success", # Assuming success if we reach this point
            components_used=component_record_ids, # The list of SmartLibrary record IDs
            output_summary=f"Form system created with ID: {form_id}. Components: {', '.join(component_record_ids)}",
            workflow_id=form_id, # Use form_id as a workflow identifier
            session_id=None, # Or a generated session ID if applicable
            tags=["form_creation", "conversational_form", form_id]
        )
        try:
            recording_result = await experience_recorder_tool.run(input=experience_input)
            if recording_result.status == "success":
                print(f"Form creation experience for {form_id} recorded successfully. Experience ID: {recording_result.experience_id}")
            else:
                # Using print for warning as basic logging setup might be out of scope for this change
                print(f"Warning: Failed to record form creation experience for {form_id}. Message: {recording_result.message}")
        except Exception as e:
            # Using print for warning
            print(f"Warning: Exception during form creation experience recording for {form_id}: {e}")

    return {
        "status": "success",
        "form_id": form_id,
        "component_record_ids": component_record_ids, # Updated key
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
    
    smart_library = SmartLibrary(container=container)
    container.register('smart_library', smart_library)
    
    # Create agent bus with null system agent
    agent_bus = SmartAgentBus(container=container)
    container.register('agent_bus', agent_bus)

    # Instantiate ExperienceRecorderTool
    experience_recorder_tool = ExperienceRecorderTool(
        agent_bus=agent_bus,
        memory_manager_agent_id=MEMORY_MANAGER_AGENT_ID
    )
    
    # Create the system agent
    system_agent = await SystemAgentFactory.create_agent(container=container)
    container.register('system_agent', system_agent)
    
    # Initialize components
    
    # Get MongoDB Client
    mongodb_client = container.get('mongodb_client')
    if not mongodb_client:
        # This part might be redundant if create_conversational_form always runs first
        # and registers the client. However, for robustness:
        # from evolving_agents.core.mongodb_client import MongoDBClient # Ensure import if not already global
        mongodb_client = MongoDBClient() 
        container.register('mongodb_client', mongodb_client)
        
    form_responses_collection_name = "eat_form_responses"
    form_definitions_collection_name = "eat_form_definitions" # Added this line
    
    await smart_library.initialize()
    await agent_bus.initialize_from_library()
    
    # Load the form definition from MongoDB
    form_definition = None
    try:
        # Ensure mongodb_client is available in this scope
        # (It should be from the previous changes to fill_out_form)
        form_definitions_collection = mongodb_client.get_collection(form_definitions_collection_name)
        form_definition = await form_definitions_collection.find_one({"form_id": form_id})
    except Exception as e:
        print(f"Error retrieving form definition {form_id} from MongoDB: {e}")
        return {
            "status": "error",
            "message": f"Database error retrieving form with ID {form_id}"
        }

    if not form_definition:
        print(f"Form definition for {form_id} not found in MongoDB collection '{form_definitions_collection_name}'.")
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
    
    # Save the responses to MongoDB
    form_responses_data = {
        "form_id": form_id,
        "responses": all_results,
        "summary": summary_result.result.text,
        "submitted_at": datetime.now(timezone.utc) # Store submission time
    }
    try:
        form_responses_collection = mongodb_client.get_collection(form_responses_collection_name)
        await form_responses_collection.replace_one(
            {"form_id": form_id}, 
            form_responses_data, 
            upsert=True
        )
        print(f"Form responses for {form_id} saved to MongoDB collection '{form_responses_collection_name}'.")
    except Exception as e:
        print(f"Error saving form responses for {form_id} to MongoDB: {e}")
        # Optionally, re-raise or handle as appropriate

    # Prepare and record the experience for form filling
    form_component_ids = form_definition.get("component_record_ids", [])
    user_responses_summary = json.dumps(user_responses)

    experience_input = ExperienceDataInput(
        primary_goal_description="User filling out a conversational form.",
        sub_task_description=f"Processing user responses for form_id: {form_id}.",
        initiating_agent_id="form_filling_service",
        final_outcome="success",
        components_used=form_component_ids,
        input_context_summary=user_responses_summary,
        output_summary=summary_result.result.text,
        workflow_id=form_id,
        session_id=None,
        tags=["form_filling", "user_response", form_id]
    )
    try:
        recording_result = await experience_recorder_tool.run(input=experience_input)
        if recording_result.status == "success":
            print(f"Form filling experience for {form_id} recorded successfully. Experience ID: {recording_result.experience_id}")
        else:
            print(f"Warning: Failed to record form filling experience for {form_id}. Message: {recording_result.message}")
    except Exception as e:
        print(f"Warning: Exception during form filling experience recording for {form_id}: {e}")
    
    return {
        "status": "success",
        "form_id": form_id,
        "response_count": len(user_responses),
        "summary": summary_result.result.text
    }

async def main():
    # Initial setup for MongoDBClient in main for verification
    # from evolving_agents.core.dependency_container import DependencyContainer # Already imported globally
    # from evolving_agents.core.mongodb_client import MongoDBClient # Already imported globally
    # from datetime import datetime, timezone # Already imported globally
    
    container = DependencyContainer()
    try:
        mongodb_client = MongoDBClient() # Assumes .env is configured
        container.register('mongodb_client', mongodb_client)
        print("MongoDBClient initialized in main for verification.")
    except Exception as e:
        print(f"Failed to initialize MongoDBClient in main for verification: {e}")
        print("Please ensure MONGODB_URI and MONGODB_DATABASE_NAME are set in your .env file.")
        mongodb_client = None # Set to None if initialization fails

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
    print(f"Component Record IDs: {form_result['component_record_ids']}")

    if mongodb_client:
        print("\n--- Verifying restaurant_feedback definition from MongoDB ---")
        try:
            form_defs_collection = mongodb_client.get_collection("eat_form_definitions")
            definition = await form_defs_collection.find_one({"form_id": "restaurant_feedback"})
            if definition:
                print("Found definition in MongoDB:")
                print(json.dumps(definition, indent=2, default=str)) # Use default=str for datetime
            else:
                print("Definition for restaurant_feedback not found in MongoDB.")
        except Exception as e:
            print(f"Error during MongoDB verification (definition): {e}")
    
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

    if mongodb_client:
        print("\n--- Verifying restaurant_feedback responses from MongoDB ---")
        try:
            form_resps_collection = mongodb_client.get_collection("eat_form_responses")
            responses_doc = await form_resps_collection.find_one({"form_id": "restaurant_feedback"})
            if responses_doc:
                print("Found responses in MongoDB:")
                # Print a summary or part of the responses to avoid too much output
                print(f"  Form ID: {responses_doc.get('form_id')}")
                print(f"  Summary: {responses_doc.get('summary')[:200]}...") # Print first 200 chars of summary
                print(f"  Response count: {len(responses_doc.get('responses', []))}")
                print(f"  Submitted At: {responses_doc.get('submitted_at')}")
            else:
                print("Responses for restaurant_feedback not found in MongoDB.")
        except Exception as e:
            print(f"Error during MongoDB verification (responses): {e}")
    
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
    print("Form definitions and responses are now stored in MongoDB.")
    print("Relevant MongoDB collections (config dependent, typically):")
    print(f"  - Form Definitions: eat_form_definitions")
    print(f"  - Form Responses: eat_form_responses")

if __name__ == "__main__":
    asyncio.run(main())