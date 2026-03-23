import pandas as pd
import random

df = pd.read_csv("data/loan_data.csv")

rows = []

genders = ["Male", "Female"]
married_opts = ["Yes", "No"]
dependents_opts = ["0", "1", "2", "3+"]
education_opts = ["Graduate", "Not Graduate"]
self_employed_opts = ["Yes", "No"]
areas = ["Urban", "Semiurban", "Rural", "None"]

for i in range(400):

    gender = random.choice(genders)
    married = random.choices(married_opts, weights=[0.65, 0.35])[0]
    dependents = random.choices(dependents_opts, weights=[0.4,0.3,0.2,0.1])[0]
    education = random.choices(education_opts, weights=[0.7,0.3])[0]
    self_employed = random.choices(self_employed_opts, weights=[0.2,0.8])[0]
    area = random.choices(areas, weights=[0.35,0.3,0.25,0.1])[0]

    if area == "Urban":
        base_income = random.randint(30000, 120000)
    elif area == "Semiurban":
        base_income = random.randint(20000, 80000)
    elif area == "Rural":
        base_income = random.randint(10000, 50000)
    else: 
        base_income = random.randint(15000, 60000)

    if education == "Graduate":
        base_income *= random.uniform(1.2, 1.6)

    if self_employed == "Yes":
        base_income *= random.uniform(0.8, 1.5)

    applicant_income = int(base_income)

    if married == "Yes":
        coapplicant_income = int(random.uniform(0.2, 0.7) * applicant_income)
    else:
        coapplicant_income = int(random.uniform(0, 0.3) * applicant_income)

    total_income = applicant_income + coapplicant_income

    loan_amount = int(total_income * random.uniform(0.2, 0.6) / 1000)

    loan_term = random.choices([360, 180, 240], weights=[0.7,0.2,0.1])[0]

    credit_history = random.choices([1,0], weights=[0.75,0.25])[0]

    emi = loan_amount * 1000 / loan_term

    affordability = emi / (total_income + 1)

    if credit_history == 1 and affordability < 0.3:
        loan_status = "Y"
    elif credit_history == 1 and affordability < 0.5:
        loan_status = random.choices(["Y","N"], weights=[0.7,0.3])[0]
    else:
        loan_status = random.choices(["Y","N"], weights=[0.2,0.8])[0]

    rows.append({
        "gender": gender,
        "married": married,
        "dependents": dependents,
        "education": education,
        "self_employed": self_employed,
        "applicant_income": applicant_income,
        "coapplicant_income": coapplicant_income,
        "loan_amount": loan_amount,
        "loan_term": loan_term,
        "credit_history": credit_history,
        "property_area": area,
        "loan_status": loan_status
    })

generated_df = pd.DataFrame(rows)

final_df = pd.concat([df, generated_df], ignore_index=True)

final_df.to_csv("data/loan_data_large.csv", index=False)

print("Dataset generated:", len(final_df), "rows")