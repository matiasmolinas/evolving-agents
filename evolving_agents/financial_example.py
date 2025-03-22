# financial_example.py
import asyncio
import os
import sys
from run_architect import run_architect_agent, initialize_system, print_step

# Define sample invoice data
SAMPLE_INVOICE = """
INVOICE #12345
Date: 2023-05-15
Vendor: TechSupplies Inc.
Address: 123 Tech Blvd, San Francisco, CA 94107

Bill To:
Acme Corporation
456 Business Ave
New York, NY 10001

Items:
1. Laptop Computer - $1,200.00 (2 units)
2. External Monitor - $300.00 (3 units)
3. Wireless Keyboard - $50.00 (5 units)

Subtotal: $2,950.00
Tax (8.5%): $250.75
Total Due: $3,200.75

Payment Terms: Net 30
Due Date: 2023-06-14

Thank you for your business!
"""

async def main():
    print_step("FINANCIAL INVOICE PROCESSING EXAMPLE", 
             "This demonstration shows how specialized agents can collaborate to process invoices", 
             "INFO")
    
    # Create data directory if it doesn't exist
    os.makedirs("data", exist_ok=True)
    
    # Check if the library file exists
    if not os.path.exists("data/financial_library.json"):
        print_step("MISSING LIBRARY FILE", 
                 "Please run generate_smart_library.py first to create the required library files", 
                 "ERROR")
        sys.exit(1)
    
    # Initialize the system with the financial library
    system = await initialize_system("data/financial_library.json")
    
    # Create prompts directory if it doesn't exist
    os.makedirs("prompts", exist_ok=True)
    
    # Check if the prompt file exists
    if not os.path.exists("prompts/financial_prompt.txt"):
        print_step("MISSING PROMPT FILE", 
                 "Please create the financial prompt file at prompts/financial_prompt.txt", 
                 "ERROR")
        sys.exit(1)
    
    # Run the architect agent with the financial prompt
    result = await run_architect_agent(
        system,
        "prompts/financial_prompt.txt",
        SAMPLE_INVOICE,
        "financial"
    )
    
    print_step("FINANCIAL EXAMPLE COMPLETED", 
             "The financial invoice processing system has been designed and executed", 
             "INFO")

if __name__ == "__main__":
    asyncio.run(main())