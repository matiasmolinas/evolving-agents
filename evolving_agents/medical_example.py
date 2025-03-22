# medical_example.py
import asyncio
import os
import sys
from run_architect import run_architect_agent, initialize_system, print_step

# Define sample patient data
SAMPLE_PATIENT_DATA = """
PATIENT CARDIOVASCULAR ASSESSMENT
Patient ID: P1293847
Date: 2023-09-15
Age: 58
Sex: Male
Height: 175 cm
Weight: 88 kg

VITAL SIGNS:
Resting Heart Rate: 78 bpm
Blood Pressure (Systolic/Diastolic): 148/92 mmHg
Respiratory Rate: 16 breaths/min
Oxygen Saturation: 96%
Temperature: 37.1°C

LAB RESULTS:
Total Cholesterol: 235 mg/dL
HDL Cholesterol: 38 mg/dL
LDL Cholesterol: 165 mg/dL
Triglycerides: 190 mg/dL
Fasting Glucose: 108 mg/dL
HbA1c: 5.9%

HISTORY:
Family History: Father had MI at age 62, Mother with hypertension
Smoking Status: Former smoker (quit 5 years ago, 20 pack-years)
Physical Activity: Sedentary (less than 30 min exercise per week)
Current Medications: Lisinopril 10mg daily

SYMPTOMS:
Occasional chest discomfort during exertion
Mild shortness of breath climbing stairs
Fatigue in the afternoons
No syncope or palpitations
"""

async def main():
    print_step("MEDICAL ASSESSMENT EXAMPLE", 
             "This demonstration shows how specialized agents can collaborate for medical assessment", 
             "INFO")
    
    # Create data directory if it doesn't exist
    os.makedirs("data", exist_ok=True)
    
    # Check if the library file exists
    if not os.path.exists("data/medical_library.json"):
        print_step("MISSING LIBRARY FILE", 
                 "Please run generate_smart_library.py first to create the required library files", 
                 "ERROR")
        sys.exit(1)
    
    # Initialize the system with the medical library
    system = await initialize_system("data/medical_library.json")
    
    # Create prompts directory if it doesn't exist
    os.makedirs("prompts", exist_ok=True)
    
    # Check if the prompt file exists
    if not os.path.exists("prompts/medical_prompt.txt"):
        print_step("MISSING PROMPT FILE", 
                 "Please create the medical prompt file at prompts/medical_prompt.txt", 
                 "ERROR")
        sys.exit(1)
    
    # Run the architect agent with the medical prompt
    result = await run_architect_agent(
        system,
        "prompts/medical_prompt.txt",
        SAMPLE_PATIENT_DATA,
        "medical"
    )
    
    print_step("MEDICAL EXAMPLE COMPLETED", 
             "The cardiovascular assessment system has been designed and executed", 
             "INFO")

if __name__ == "__main__":
    asyncio.run(main())