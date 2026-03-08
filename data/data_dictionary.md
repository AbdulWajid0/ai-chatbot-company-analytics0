# Data Dictionary

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
