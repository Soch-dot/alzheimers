# Alzheimer's Dataset Description

## Column Explanations

### **age** (numeric)
- Patient's age in years
- Range: typically 65-85 for Alzheimer's studies

### **gender** (categorical: M or F)
- Male (M) or Female (F)
- We'll convert this to numbers later (0 or 1)

### **mmse_score** (numeric: 0-30)
- **Mini-Mental State Examination** score
- Higher = better cognitive function
- 24-30 = Normal
- 18-23 = Mild cognitive impairment
- 0-17 = Severe cognitive impairment

### **cdr_score** (numeric: 0, 0.5, 1.0, 1.5, 2.0)
- **Clinical Dementia Rating** scale
- 0 = No dementia
- 0.5 = Questionable dementia
- 1.0 = Mild dementia
- 1.5 = Moderate dementia
- 2.0 = Severe dementia

### **education_years** (numeric)
- Years of formal education
- Can be a protective factor

### **apoe4** (binary: 0 or 1)
- **APOE ε4 allele** presence
- 1 = Has the gene variant (higher risk)
- 0 = Does not have it
- Major genetic risk factor for Alzheimer's

### **family_history** (binary: 0 or 1)
- Family history of Alzheimer's
- 1 = Yes, has family history
- 0 = No family history

### **diagnosis** (binary: 0 or 1) - **THIS IS OUR TARGET!**
- 0 = No Alzheimer's (healthy)
- 1 = Has Alzheimer's
- This is what we want to predict!

## What We'll Do

1. Use age, gender, test scores, etc. (features) to predict diagnosis (target)
2. Train machine learning models to learn patterns
3. Evaluate how well our models predict Alzheimer's risk

