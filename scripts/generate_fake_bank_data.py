import pandas as pd
import random
from faker import Faker
from datetime import datetime, timedelta
from pathlib import Path
import zipfile

# ===========================================
# CONFIGURATION
# ===========================================
NUM_RECORDS = 5000
OUT_DIR = Path("data/bank_documents")
OUT_DIR.mkdir(exist_ok=True, parents=True)

fake = Faker()
Faker.seed(42)
random.seed(42)

# ===========================================
# 1️⃣ Generate FAQ Data
# ===========================================
topics = [
    "checking account", "savings account", "credit card", "auto loan", 
    "mortgage", "wire transfer", "Zelle payment", "overdraft", 
    "mobile deposit", "fraud alert", "account security", "debit card"
]

faq_data = []
for i in range(NUM_RECORDS):
    topic = random.choice(topics)
    q = f"How do I {random.choice(['open', 'close', 'use', 'manage', 'update'])} my {topic}?"
    a = f"To {q.split()[2]} your {topic}, log in to your online banking portal or visit your nearest branch with valid identification."
    faq_data.append((q, a))

df_faq = pd.DataFrame(faq_data, columns=["question", "answer"])
df_faq.to_csv(OUT_DIR / "faqs_fake_bank.csv", index=False)
print(f"✅ Generated {len(df_faq)} FAQ records")

# ===========================================
# 2️⃣ Generate Transactions
# ===========================================
categories = ["Grocery Store", "ATM Withdrawal", "Online Payment", "Netflix", "Utility Bill", "Gas Station", "Restaurant", "Pharmacy", "Amazon", "Rent Payment"]
start_date = datetime(2023, 1, 1)

transactions = []
for i in range(NUM_RECORDS):
    date = (start_date + timedelta(days=random.randint(0, 365))).strftime("%m/%d/%Y")
    desc = random.choice(categories)
    ttype = random.choice(["debit", "credit"])
    amount = round(random.uniform(5, 1500), 2)
    balance = round(random.uniform(50, 20000), 2)
    transactions.append({
        "date": date,
        "description": desc,
        "type": ttype,
        "amount": amount,
        "balance_after": balance
    })

df_tx = pd.DataFrame(transactions)
df_tx.to_csv(OUT_DIR / "fake_statement.csv", index=False)
print(f"✅ Generated {len(df_tx)} transaction records")

# ===========================================
# 3️⃣ Generate Customer Profiles
# ===========================================
account_types = ["Checking", "Savings", "Credit Card", "Money Market", "CD"]
customers = []

for i in range(NUM_RECORDS):
    customers.append({
        "customer_id": fake.uuid4(),
        "full_name": fake.name(),
        "email": fake.email(),
        "phone": fake.phone_number(),
        "account_type": random.choice(account_types),
        "balance": round(random.uniform(10, 50000), 2),
        "city": fake.city(),
        "state": fake.state_abbr(),
        "zip": fake.zipcode(),
        "created_on": fake.date_between(start_date='-5y', end_date='today')
    })

df_cust = pd.DataFrame(customers)
df_cust.to_csv(OUT_DIR / "fake_customers.csv", index=False)
print(f"✅ Generated {len(df_cust)} customer records")

print(f"📦 All files saved to: {OUT_DIR.resolve()}")
