import pandas as pd
import random

# load original dataset
df = pd.read_csv("data/loan_data.csv")

rows = []

genders = ["Male", "Female"]
married = ["Yes", "No"]
dependents = ["0", "1", "2", "3+"]
education = ["Graduate", "Not Graduate"]
self_employed = ["Yes", "No"]
areas = ["Urban", "Semiurban", "Rural"]

for i in range(400):

    income = random.randint(2000,15000)
    co_income = random.randint(0,5000)
    loan_amount = random.randint(80,350)

    credit_history = random.choice([0,1])

    # simple rule for realistic approval
    if credit_history == 1 and income > 3000:
        loan_status = "Y"
    else:
        loan_status = random.choice(["Y","N"])

    rows.append({
        "gender": random.choice(genders),
        "married": random.choice(married),
        "dependents": random.choice(dependents),
        "education": random.choice(education),
        "self_employed": random.choice(self_employed),
        "applicant_income": income,
        "coapplicant_income": co_income,
        "loan_amount": loan_amount,
        "loan_term": 360,
        "credit_history": credit_history,
        "property_area": random.choice(areas),
        "loan_status": loan_status
    })

generated_df = pd.DataFrame(rows)

# combine original + generated
final_df = pd.concat([df, generated_df], ignore_index=True)

# save new dataset
final_df.to_csv("data/loan_data_large.csv", index=False)

print("Dataset generated:", len(final_df), "rows")