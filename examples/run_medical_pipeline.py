import asyncio
from evolving_agents.agents import create_architect_zero
from evolving_agents.smart_library.smart_library import SmartLibrary
from evolving_agents.core.system_agent import SystemAgentFactory
from evolving_agents.core.llm_service import LLMService
from evolving_agents.agent_bus.smart_agent_bus import SmartAgentBus
import json


async def run_pipeline(prompt: str, input_path: str, output_path: str):
    # Step 1: Load input .txt
    with open(input_path, "r") as f:
        report_content = f.read()

    # Step 2: Set up services
    llm_service = LLMService(...)  # Your OpenAI/GPT service wrapper
    smart_library = SmartLibrary("smart_library.json")

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

    # Step 3: Create the architect agent
    architect = await create_architect_zero(
        llm_service=llm_service,
        smart_library=smart_library,
        agent_bus=agent_bus,
        system_agent_factory=SystemAgentFactory.create_agent
    )

    # Step 4: Run the full task by passing prompt and report
    full_prompt = (
        f"{prompt}\n\n"
        f"Here is the content of the medical report:\n\n"
        f"{report_content}"
    )

    result = await architect.run(full_prompt)

    # Step 5: Save the result
    with open(output_path, "w") as f:
        f.write(result)

    print("Pipeline completed. Output written to", output_path)


# ---- MAIN ----
if __name__ == "__main__":
    asyncio.run(run_pipeline(
        prompt="Build a system that reads a medical report from a .txt file, extracts the patient diagnosis and medications, and generates a short summary saved in a .txt file.",
        input_path="medical_report.txt",
        output_path="summary_output.txt"
    ))
