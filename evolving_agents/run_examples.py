# run_examples.py
import asyncio
import os
import subprocess
import sys

async def main():
    # Check if we need to generate libraries first
    if not os.path.exists("data/financial_library.json") or not os.path.exists("data/medical_library.json"):
        print("Generating library files...")
        subprocess.run([sys.executable, "generate_smart_library.py"])
    
    # Create prompts directory if it doesn't exist
    os.makedirs("prompts", exist_ok=True)
    
    # Create prompt files if they don't exist
    if not os.path.exists("prompts/financial_prompt.txt"):
        print("Creating financial prompt file...")
        with open("prompts/financial_prompt.txt", "w") as f:
            f.write("""Create an advanced invoice processing system that improves upon the basic version in the library. The system should:

1. Use a more sophisticated document analyzer that can detect invoices with higher confidence
2. Extract comprehensive information (invoice number, date, vendor, items, subtotal, tax, total)
3. Verify calculations to ensure subtotal + tax = total
4. Generate a structured summary with key insights
5. Handle different invoice formats and detect potential errors

The system should leverage existing components from the library when possible,
evolve them where improvements are needed, and create new components for missing functionality.

Please generate a complete workflow for this invoice processing system.""")
    
    if not os.path.exists("prompts/medical_prompt.txt"):
        print("Creating medical prompt file...")
        with open("prompts/medical_prompt.txt", "w") as f:
            f.write("""Create a cardiovascular health assessment system that analyzes patient physiological data to evaluate cardiovascular disease risk. 
The system should:

1. Extract physiological data from patient records (vital signs, lab values, anthropometrics, risk factors)
2. Calculate Body Mass Index (BMI) and classify weight status
3. Apply the Framingham Risk Score formula to determine 10-year cardiovascular disease risk
4. Analyze the combined data to provide clinical interpretations, concerning findings, and evidence-based recommendations
5. Generate a comprehensive patient assessment with appropriate medical disclaimers

The system should use established medical formulas and guidelines for all calculations and risk assessments.
Leverage existing components from the library when possible and create new components for missing functionality.

Please generate a complete workflow for this cardiovascular assessment system.""")
    
    # Run financial example
    print("\n\n======== RUNNING FINANCIAL EXAMPLE ========\n\n")
    result_fin = subprocess.run([sys.executable, "financial_example.py"])
    
    # Run medical example
    print("\n\n======== RUNNING MEDICAL EXAMPLE ========\n\n")
    result_med = subprocess.run([sys.executable, "medical_example.py"])
    
    print("\n\nAll examples completed. Results saved in respective output files.")

if __name__ == "__main__":
    asyncio.run(main())