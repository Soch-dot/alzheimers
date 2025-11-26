"""
LESSON 1: Python Basics - Variables, Data Types, Lists, Dictionaries
====================================================================

Welcome to your first lesson! Today we'll learn the building blocks of Python
that you'll use throughout your Alzheimer's prediction project.

WHAT YOU'LL LEARN:
- Variables (storing data)
- Data types (numbers, text, True/False)
- Lists (storing multiple values)
- Dictionaries (storing labeled data)

CONNECTION TO ALZHEIMER'S PROJECT:
- We'll work with patient data like age, test scores, and diagnosis
- Lists will store multiple patients
- Dictionaries will store all info about one patient
"""

print("=" * 60)
print("LESSON 1: Python Basics")
print("=" * 60)
print()

# ============================================================================
# PART 1: VARIABLES - Storing single pieces of data
# ============================================================================

print("PART 1: Variables")
print("-" * 40)

# Variables are like labeled boxes that store data
# In our Alzheimer's project, we'll store patient information

# Example: A single patient's age
patient_age = 72
print(f"Patient age: {patient_age}")

# Example: A patient's MMSE score (cognitive test)
mmse_score = 24
print(f"MMSE Score: {mmse_score}")

# Example: Patient's gender
patient_gender = "M"  # M for Male, F for Female
print(f"Gender: {patient_gender}")

# Example: Does patient have Alzheimer's? (True or False)
has_alzheimers = True
print(f"Has Alzheimer's: {has_alzheimers}")

print()

# ============================================================================
# PART 2: DATA TYPES - Different kinds of data
# ============================================================================

print("PART 2: Data Types")
print("-" * 40)

# Integer (whole numbers) - for age, scores
age = 75
print(f"Age (integer): {age}, type: {type(age)}")

# Float (decimal numbers) - for CDR score
cdr_score = 0.5
print(f"CDR Score (float): {cdr_score}, type: {type(cdr_score)}")

# String (text) - for names, categories
gender = "F"
print(f"Gender (string): {gender}, type: {type(gender)}")

# Boolean (True/False) - for yes/no questions
has_apoe4_gene = True
print(f"Has APOE4 gene (boolean): {has_apoe4_gene}, type: {type(has_apoe4_gene)}")

print()

# ============================================================================
# PART 3: LISTS - Storing multiple values in order
# ============================================================================

print("PART 3: Lists - Multiple Patients")
print("-" * 40)

# A list stores multiple values in order
# Think of it like a row in a spreadsheet

# Example: Ages of 5 patients
patient_ages = [72, 68, 75, 65, 80]
print(f"Ages of 5 patients: {patient_ages}")

# Example: MMSE scores of 5 patients
mmse_scores = [24, 28, 20, 30, 18]
print(f"MMSE scores: {mmse_scores}")

# Accessing individual items (Python counts from 0!)
print(f"First patient's age: {patient_ages[0]}")
print(f"Third patient's age: {patient_ages[2]}")
print(f"Last patient's age: {patient_ages[4]}")

# Getting the length (how many patients)
print(f"Number of patients: {len(patient_ages)}")

print()

# ============================================================================
# PART 4: DICTIONARIES - Storing labeled data (like a patient record)
# ============================================================================

print("PART 4: Dictionaries - One Patient's Complete Record")
print("-" * 40)

# A dictionary stores data with labels (keys)
# Perfect for storing all info about one patient!

# Example: One patient's complete record
patient_1 = {
    "age": 72,
    "gender": "M",
    "mmse_score": 24,
    "cdr_score": 0.5,
    "education_years": 12,
    "apoe4": 0,  # 0 = no, 1 = yes
    "family_history": 1,  # 0 = no, 1 = yes
    "diagnosis": 1  # 0 = healthy, 1 = Alzheimer's
}

print("Patient 1's complete record:")
print(patient_1)

# Accessing specific information
print(f"\nPatient 1's age: {patient_1['age']}")
print(f"Patient 1's MMSE score: {patient_1['mmse_score']}")
print(f"Patient 1's diagnosis: {patient_1['diagnosis']} (1 = Alzheimer's)")

# Another patient
patient_2 = {
    "age": 68,
    "gender": "F",
    "mmse_score": 28,
    "cdr_score": 0,
    "education_years": 16,
    "apoe4": 1,
    "family_history": 0,
    "diagnosis": 0  # Healthy!
}

print(f"\nPatient 2's age: {patient_2['age']}")
print(f"Patient 2's diagnosis: {patient_2['diagnosis']} (0 = Healthy)")

print()

# ============================================================================
# PART 5: LIST OF DICTIONARIES - Multiple patients!
# ============================================================================

print("PART 5: List of Dictionaries - Multiple Patients")
print("-" * 40)

# We can put multiple patient dictionaries in a list
# This is like having a whole dataset!

all_patients = [
    patient_1,
    patient_2,
    {
        "age": 75,
        "gender": "M",
        "mmse_score": 20,
        "cdr_score": 1.0,
        "education_years": 10,
        "apoe4": 1,
        "family_history": 1,
        "diagnosis": 1
    }
]

print(f"Total patients in dataset: {len(all_patients)}")
print(f"\nFirst patient: {all_patients[0]}")
print(f"Second patient's age: {all_patients[1]['age']}")

print()

# ============================================================================
# PRACTICE EXERCISES
# ============================================================================

print("=" * 60)
print("PRACTICE EXERCISES")
print("=" * 60)
print()
print("Try these on your own:")
print()
print("1. Create a variable called 'my_age' and set it to your age")
print("2. Create a list called 'test_scores' with values [25, 30, 22, 28]")
print("3. Create a dictionary for a new patient with:")
print("   - age: 70")
print("   - mmse_score: 26")
print("   - diagnosis: 0")
print("4. Print the second test score from your list")
print("5. Print the age from your patient dictionary")
print()
print("After you try, we'll move to Lesson 2: Functions and Loops!")
print("=" * 60)

