"""
Data Generation Script
======================
Generates sample sales, HR, and finance datasets for the AI BI Chatbot.

Run: python generate_data.py
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
from pathlib import Path

# Configuration
np.random.seed(42)
random.seed(42)

DATA_DIR = Path("data/raw")
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Date range
START_DATE = datetime(2023, 1, 1)
END_DATE = datetime(2024, 12, 31)
date_range = pd.date_range(START_DATE, END_DATE, freq='D')


def generate_sales_data(num_records=5000):
    """Generate synthetic sales data."""
    print("📊 Generating sales data...")
    
    products = [
        "Laptop Pro", "Smartphone X", "Tablet Plus", "Wireless Headphones",
        "Smart Watch", "Desktop Computer", "Monitor 4K", "Keyboard Mechanical",
        "Mouse Gaming", "External SSD", "Webcam HD", "Printer Laser"
    ]
    
    categories = ["Electronics", "Computers", "Accessories", "Audio", "Peripherals"]
    regions = ["North", "South", "East", "West", "Central"]
    channels = ["Online", "Retail", "Partner", "Direct"]
    
    data = []
    
    for _ in range(num_records):
        date = random.choice(date_range)
        product = random.choice(products)
        category = random.choice(categories)
        region = random.choice(regions)
        channel = random.choice(channels)
        
        # Pricing with some variation
        base_price = random.uniform(50, 2000)
        quantity = random.randint(1, 100)
        revenue = base_price * quantity
        
        # Profit margin between 10% and 40%
        profit_margin = random.uniform(10, 40)
        profit = revenue * (profit_margin / 100)
        
        data.append({
            'date': date,
            'product': product,
            'category': category,
            'region': region,
            'channel': channel,
            'quantity': quantity,
            'revenue': round(revenue, 2),
            'profit': round(profit, 2),
            'profit_margin': round(profit_margin, 2)
        })
    
    df = pd.DataFrame(data)
    df = df.sort_values('date')
    
    # Add computed columns
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['quarter'] = df['date'].dt.quarter
    df['year_month'] = df['date'].dt.to_period('M').astype(str)
    
    filepath = DATA_DIR / "sales_data.csv"
    df.to_csv(filepath, index=False)
    print(f"✅ Sales data saved: {filepath} ({len(df):,} records)")
    
    return df


def generate_hr_data(num_employees=500):
    """Generate synthetic HR data."""
    print("👥 Generating HR data...")
    
    departments = ["Engineering", "Sales", "Marketing", "Finance", "HR", "Operations", "Product"]
    positions = ["Manager", "Senior", "Mid-Level", "Junior", "Intern"]
    locations = ["New York", "San Francisco", "Chicago", "Austin", "Boston"]
    
    data = []
    
    for emp_id in range(1, num_employees + 1):
        department = random.choice(departments)
        position = random.choice(positions)
        location = random.choice(locations)
        
        # Salary based on position
        salary_ranges = {
            "Manager": (100000, 180000),
            "Senior": (80000, 140000),
            "Mid-Level": (60000, 100000),
            "Junior": (45000, 70000),
            "Intern": (30000, 50000)
        }
        
        min_sal, max_sal = salary_ranges[position]
        salary = random.randint(min_sal, max_sal)
        
        # Performance score (1-5)
        performance_score = round(random.uniform(2.5, 5.0), 1)
        
        # Hire date
        hire_date = START_DATE + timedelta(days=random.randint(0, 1095))  # Up to 3 years ago
        
        # Attrition (1 = left, 0 = active)
        attrition = 1 if random.random() < 0.15 else 0  # 15% attrition rate
        
        data.append({
            'employee_id': f"EMP{emp_id:05d}",
            'name': f"Employee {emp_id}",
            'department': department,
            'position': position,
            'location': location,
            'hire_date': hire_date,
            'salary': salary,
            'performance_score': performance_score,
            'attrition': attrition
        })
    
    df = pd.DataFrame(data)
    
    filepath = DATA_DIR / "hr_data.csv"
    df.to_csv(filepath, index=False)
    print(f"✅ HR data saved: {filepath} ({len(df):,} records)")
    
    return df


def generate_finance_data():
    """Generate synthetic finance data."""
    print("💰 Generating finance data...")
    
    departments = ["Engineering", "Sales", "Marketing", "Finance", "HR", "Operations", "Product"]
    
    data = []
    
    for year in [2023, 2024]:
        for quarter in range(1, 5):
            for department in departments:
                # Budget allocation
                base_budget = random.randint(100000, 1000000)
                budget = base_budget
                
                # Actual spending with some variance
                variance_pct = random.uniform(-0.15, 0.10)  # -15% to +10%
                actual = budget * (1 + variance_pct)
                variance = actual - budget
                
                data.append({
                    'year': year,
                    'quarter': quarter,
                    'department': department,
                    'budget': round(budget, 2),
                    'actual': round(actual, 2),
                    'variance': round(variance, 2),
                    'variance_pct': round(variance_pct * 100, 2)
                })
    
    df = pd.DataFrame(data)
    
    filepath = DATA_DIR / "finance_data.csv"
    df.to_csv(filepath, index=False)
    print(f"✅ Finance data saved: {filepath} ({len(df):,} records)")
    
    return df


def main():
    """Generate all datasets."""
    print("=" * 60)
    print("🚀 Generating Sample Data for AI BI Chatbot")
    print("=" * 60)
    print()
    
    # Generate datasets
    sales_df = generate_sales_data(num_records=5000)
    hr_df = generate_hr_data(num_employees=500)
    finance_df = generate_finance_data()
    
    print()
    print("=" * 60)
    print("✅ Data generation complete!")
    print("=" * 60)
    print()
    print("📊 Summary:")
    print(f"   - Sales records: {len(sales_df):,}")
    print(f"   - HR records: {len(hr_df):,}")
    print(f"   - Finance records: {len(finance_df):,}")
    print()
    print("▶️  Next step: Run 'streamlit run app.py' to start the chatbot")
    print()


if __name__ == "__main__":
    main()
