"""
Synthetic Company Data Generator
=================================
Generates realistic sample datasets for Sales, HR, and Finance.
Run this script once to populate the data/raw/ directory.

Usage:
    python generate_data.py
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import random

# Seed for reproducibility
np.random.seed(42)
random.seed(42)

# === OUTPUT DIRECTORY ===
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
os.makedirs(RAW_DIR, exist_ok=True)


def generate_sales_data(num_records=5000):
    """
    Generate synthetic sales data spanning 2 years.
    Columns: date, product, category, region, quantity, unit_price, revenue, cost, profit
    """
    print("📊 Generating Sales Data...")

    products = {
        "Laptop Pro": {"category": "Electronics", "price_range": (800, 1500), "cost_pct": 0.65},
        "Wireless Mouse": {"category": "Electronics", "price_range": (15, 45), "cost_pct": 0.40},
        "Mechanical Keyboard": {"category": "Electronics", "price_range": (60, 150), "cost_pct": 0.50},
        "USB-C Hub": {"category": "Electronics", "price_range": (25, 75), "cost_pct": 0.45},
        "Monitor 27inch": {"category": "Electronics", "price_range": (250, 500), "cost_pct": 0.60},
        "Office Chair": {"category": "Furniture", "price_range": (150, 400), "cost_pct": 0.55},
        "Standing Desk": {"category": "Furniture", "price_range": (300, 700), "cost_pct": 0.58},
        "Desk Lamp": {"category": "Furniture", "price_range": (20, 60), "cost_pct": 0.35},
        "Webcam HD": {"category": "Accessories", "price_range": (40, 120), "cost_pct": 0.42},
        "Headset Pro": {"category": "Accessories", "price_range": (50, 200), "cost_pct": 0.48},
        "Notebook Pack": {"category": "Stationery", "price_range": (5, 20), "cost_pct": 0.30},
        "Printer Paper": {"category": "Stationery", "price_range": (8, 25), "cost_pct": 0.25},
        "Whiteboard": {"category": "Stationery", "price_range": (30, 80), "cost_pct": 0.40},
        "Coffee Machine": {"category": "Appliances", "price_range": (100, 350), "cost_pct": 0.52},
        "Water Purifier": {"category": "Appliances", "price_range": (150, 400), "cost_pct": 0.55},
    }

    regions = ["North", "South", "East", "West", "Central"]
    sales_channels = ["Online", "Retail", "Wholesale", "B2B"]

    # Date range: Jan 2024 to Dec 2025
    start_date = datetime(2024, 1, 1)
    end_date = datetime(2025, 12, 31)
    date_range = (end_date - start_date).days

    records = []
    for _ in range(num_records):
        # Random date with seasonal bias (more sales in Q4)
        day_offset = np.random.randint(0, date_range)
        date = start_date + timedelta(days=day_offset)

        # Seasonal multiplier (higher sales in Q4)
        month = date.month
        seasonal_mult = 1.0
        if month in [11, 12]:
            seasonal_mult = 1.4
        elif month in [1, 2]:
            seasonal_mult = 0.8
        elif month in [6, 7]:
            seasonal_mult = 1.15

        # Pick product
        product_name = random.choice(list(products.keys()))
        product_info = products[product_name]

        # Quantity with seasonal adjustment
        base_qty = np.random.randint(1, 25)
        quantity = max(1, int(base_qty * seasonal_mult))

        # Price with some variance
        min_price, max_price = product_info["price_range"]
        unit_price = round(np.random.uniform(min_price, max_price), 2)

        revenue = round(quantity * unit_price, 2)
        cost = round(revenue * product_info["cost_pct"], 2)
        profit = round(revenue - cost, 2)

        records.append({
            "date": date.strftime("%Y-%m-%d"),
            "product": product_name,
            "category": product_info["category"],
            "region": random.choice(regions),
            "sales_channel": random.choice(sales_channels),
            "quantity": quantity,
            "unit_price": unit_price,
            "revenue": revenue,
            "cost": cost,
            "profit": profit,
        })

    df = pd.DataFrame(records)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    # Add some realistic missing values (2% of data)
    mask = np.random.random(len(df)) < 0.02
    df.loc[mask, "sales_channel"] = np.nan

    filepath = os.path.join(RAW_DIR, "sales_data.csv")
    df.to_csv(filepath, index=False)
    print(f"   ✅ Sales data: {len(df)} records → {filepath}")
    return df


def generate_hr_data(num_employees=500):
    """
    Generate synthetic HR/employee data.
    Columns: employee_id, name, department, designation, salary, join_date,
             age, gender, performance_score, attrition
    """
    print("👥 Generating HR Data...")

    departments = ["Engineering", "Sales", "Marketing", "HR", "Finance", "Operations", "Support"]
    designations = {
        "Engineering": ["Junior Developer", "Senior Developer", "Tech Lead", "Engineering Manager"],
        "Sales": ["Sales Associate", "Sales Executive", "Sales Manager", "VP Sales"],
        "Marketing": ["Marketing Analyst", "Content Specialist", "Marketing Manager", "CMO"],
        "HR": ["HR Associate", "HR Specialist", "HR Manager", "HR Director"],
        "Finance": ["Accountant", "Financial Analyst", "Finance Manager", "CFO"],
        "Operations": ["Operations Associate", "Operations Lead", "Operations Manager", "COO"],
        "Support": ["Support Associate", "Support Lead", "Support Manager", "Support Director"],
    }
    salary_ranges = {
        "Junior Developer": (40000, 65000), "Senior Developer": (65000, 100000),
        "Tech Lead": (90000, 130000), "Engineering Manager": (120000, 170000),
        "Sales Associate": (30000, 50000), "Sales Executive": (50000, 80000),
        "Sales Manager": (80000, 120000), "VP Sales": (130000, 180000),
        "Marketing Analyst": (35000, 55000), "Content Specialist": (40000, 60000),
        "Marketing Manager": (75000, 110000), "CMO": (140000, 200000),
        "HR Associate": (30000, 45000), "HR Specialist": (45000, 65000),
        "HR Manager": (70000, 100000), "HR Director": (110000, 150000),
        "Accountant": (35000, 55000), "Financial Analyst": (50000, 80000),
        "Finance Manager": (85000, 120000), "CFO": (150000, 220000),
        "Operations Associate": (30000, 45000), "Operations Lead": (50000, 70000),
        "Operations Manager": (75000, 110000), "COO": (140000, 200000),
        "Support Associate": (28000, 40000), "Support Lead": (40000, 60000),
        "Support Manager": (65000, 95000), "Support Director": (100000, 140000),
    }
    first_names = ["James", "Mary", "John", "Patricia", "Robert", "Jennifer", "Michael", "Linda",
                   "David", "Elizabeth", "William", "Barbara", "Richard", "Susan", "Joseph", "Jessica",
                   "Thomas", "Sarah", "Christopher", "Karen", "Daniel", "Lisa", "Matthew", "Nancy",
                   "Aisha", "Raj", "Priya", "Omar", "Fatima", "Chen", "Yuki", "Sofia",
                   "Ahmed", "Maria", "Li", "Emma", "Noah", "Olivia", "Liam", "Ava"]
    last_names = ["Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller", "Davis",
                  "Rodriguez", "Martinez", "Hernandez", "Lopez", "Gonzalez", "Wilson", "Anderson",
                  "Thomas", "Taylor", "Moore", "Jackson", "Martin", "Patel", "Khan", "Ali",
                  "Singh", "Kumar", "Lee", "Wang", "Chen", "Kim", "Nakamura"]

    records = []
    for i in range(1, num_employees + 1):
        dept = random.choice(departments)
        desig = random.choice(designations[dept])
        sal_min, sal_max = salary_ranges[desig]

        # Join date (within last 10 years)
        join_offset = np.random.randint(0, 3650)
        join_date = datetime(2026, 3, 1) - timedelta(days=join_offset)

        # Age based on designation seniority
        if "Junior" in desig or "Associate" in desig:
            age = np.random.randint(22, 32)
        elif "Senior" in desig or "Lead" in desig or "Specialist" in desig or "Analyst" in desig or "Executive" in desig:
            age = np.random.randint(28, 42)
        elif "Manager" in desig:
            age = np.random.randint(32, 50)
        else:
            age = np.random.randint(38, 58)

        # Performance score (1-5, normally distributed around 3.5)
        perf_score = min(5, max(1, round(np.random.normal(3.5, 0.8))))

        # Attrition probability (higher for low performers and low salary relative to role)
        salary = round(np.random.uniform(sal_min, sal_max), 2)
        attrition_prob = 0.12  # base rate
        if perf_score <= 2:
            attrition_prob += 0.15
        if salary < (sal_min + sal_max) / 2 * 0.8:
            attrition_prob += 0.08
        attrition = 1 if np.random.random() < attrition_prob else 0

        records.append({
            "employee_id": f"EMP{i:04d}",
            "name": f"{random.choice(first_names)} {random.choice(last_names)}",
            "department": dept,
            "designation": desig,
            "salary": salary,
            "join_date": join_date.strftime("%Y-%m-%d"),
            "age": age,
            "gender": random.choice(["Male", "Female", "Male", "Female", "Non-Binary"]),
            "performance_score": perf_score,
            "attrition": attrition,
        })

    df = pd.DataFrame(records)
    filepath = os.path.join(RAW_DIR, "hr_data.csv")
    df.to_csv(filepath, index=False)
    print(f"   ✅ HR data: {len(df)} records → {filepath}")
    return df


def generate_finance_data(num_months=24):
    """
    Generate synthetic company finance data (monthly).
    Columns: month, year, expense_type, amount, budget, department
    """
    print("💰 Generating Finance Data...")

    expense_types = {
        "Salaries": {"budget_range": (200000, 350000), "variance": 0.05},
        "Marketing": {"budget_range": (30000, 80000), "variance": 0.20},
        "Infrastructure": {"budget_range": (15000, 40000), "variance": 0.10},
        "R&D": {"budget_range": (50000, 120000), "variance": 0.15},
        "Travel": {"budget_range": (5000, 25000), "variance": 0.30},
        "Software Licenses": {"budget_range": (10000, 30000), "variance": 0.08},
        "Office Supplies": {"budget_range": (3000, 10000), "variance": 0.12},
        "Utilities": {"budget_range": (5000, 15000), "variance": 0.06},
        "Training": {"budget_range": (8000, 25000), "variance": 0.18},
        "Consulting": {"budget_range": (10000, 50000), "variance": 0.25},
    }

    departments = ["Engineering", "Sales", "Marketing", "HR", "Finance", "Operations"]

    records = []
    start_date = datetime(2024, 1, 1)

    for month_offset in range(num_months):
        current_date = start_date + timedelta(days=month_offset * 30)
        month = current_date.month
        year = current_date.year

        for expense_type, info in expense_types.items():
            for dept in random.sample(departments, k=random.randint(2, 4)):
                budget_min, budget_max = info["budget_range"]
                # Scale budget per department (not all get the full range)
                dept_scale = np.random.uniform(0.3, 1.0)
                budget = round(np.random.uniform(budget_min, budget_max) * dept_scale / len(departments), 2)

                # Actual amount varies from budget
                variance = info["variance"]
                actual = round(budget * np.random.uniform(1 - variance, 1 + variance * 1.5), 2)

                records.append({
                    "month": month,
                    "year": year,
                    "date": f"{year}-{month:02d}-01",
                    "expense_type": expense_type,
                    "department": dept,
                    "budget": budget,
                    "actual_amount": actual,
                    "variance": round(actual - budget, 2),
                    "variance_pct": round(((actual - budget) / budget) * 100, 2) if budget > 0 else 0,
                })

    df = pd.DataFrame(records)
    df["date"] = pd.to_datetime(df["date"])
    filepath = os.path.join(RAW_DIR, "finance_data.csv")
    df.to_csv(filepath, index=False)
    print(f"   ✅ Finance data: {len(df)} records → {filepath}")
    return df


def generate_data_dictionary():
    """Generate a data dictionary documenting all dataset columns."""
    print("📖 Generating Data Dictionary...")

    content = """# Data Dictionary

## Sales Data (`sales_data.csv`)

| Column | Type | Description |
|--------|------|-------------|
| date | datetime | Transaction date |
| product | string | Product name |
| category | string | Product category (Electronics, Furniture, Accessories, Stationery, Appliances) |
| region | string | Sales region (North, South, East, West, Central) |
| sales_channel | string | Channel (Online, Retail, Wholesale, B2B) |
| quantity | integer | Units sold |
| unit_price | float | Price per unit ($) |
| revenue | float | Total revenue (quantity × unit_price) |
| cost | float | Cost of goods sold |
| profit | float | Profit (revenue - cost) |

## HR Data (`hr_data.csv`)

| Column | Type | Description |
|--------|------|-------------|
| employee_id | string | Unique employee ID (EMP0001) |
| name | string | Employee full name |
| department | string | Department name |
| designation | string | Job title |
| salary | float | Annual salary ($) |
| join_date | datetime | Date of joining |
| age | integer | Employee age |
| gender | string | Gender |
| performance_score | integer | Performance rating (1-5) |
| attrition | integer | Left company (1=Yes, 0=No) |

## Finance Data (`finance_data.csv`)

| Column | Type | Description |
|--------|------|-------------|
| month | integer | Month number (1-12) |
| year | integer | Year |
| date | datetime | First day of month |
| expense_type | string | Type of expense |
| department | string | Department responsible |
| budget | float | Budgeted amount ($) |
| actual_amount | float | Actual spent ($) |
| variance | float | Over/under budget ($) |
| variance_pct | float | Variance as percentage |
"""
    filepath = os.path.join(BASE_DIR, "data", "data_dictionary.md")
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"   ✅ Data dictionary → {filepath}")


# === MAIN ===
if __name__ == "__main__":
    print("=" * 60)
    print("🏭 SYNTHETIC COMPANY DATA GENERATOR")
    print("=" * 60)
    print()

    sales_df = generate_sales_data(5000)
    hr_df = generate_hr_data(500)
    finance_df = generate_finance_data(24)
    generate_data_dictionary()

    print()
    print("=" * 60)
    print("✅ ALL DATA GENERATED SUCCESSFULLY!")
    print(f"   📊 Sales:   {len(sales_df)} records")
    print(f"   👥 HR:      {len(hr_df)} records")
    print(f"   💰 Finance: {len(finance_df)} records")
    print("=" * 60)
