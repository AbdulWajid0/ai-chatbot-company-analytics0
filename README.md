# 🤖 AI Chatbot with Company Data Analysis Capability

A Conversational Business Intelligence Assistant built with **Streamlit** and **Google Gemini API**.

## 📌 What This System Does

- Accepts natural language queries (e.g., *"What were last quarter's sales?"*)
- Identifies user intent and maps queries to dataset fields
- Performs dynamic data analysis (filtering, aggregation, KPIs)
- Generates interactive visualizations
- Provides summary insights and follow-up suggestions
- Generates downloadable PDF analytics reports

## 🛠 Tech Stack

| Component | Technology |
|-----------|-----------|
| Frontend | Streamlit |
| AI/NLP | Google Gemini API |
| Data Processing | Pandas, NumPy |
| Visualization | Plotly, Matplotlib, Seaborn |
| Forecasting | statsmodels (ARIMA) |
| PDF Export | FPDF2 |

## 🚀 Setup Instructions

### 1. Clone the Repository
```bash
git clone https://github.com/AbdulWajid0/ai-chatbot-company-analytics0.git
cd ai-chatbot-company-analytics0
```

### 2. Create Virtual Environment
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure API Key
1. Go to [Google AI Studio](https://aistudio.google.com/apikey)
2. Create an API key
3. Open `.env` and replace `your_api_key_here_replace_me` with your key

### 5. Run the App
```bash
streamlit run app.py
```

## 📂 Project Structure

```
ai-chatbot-company-analytics/
├── app.py                      # Main Streamlit chatbot app
├── requirements.txt            # Dependencies
├── .env                        # API key (not committed)
├── data/
│   ├── raw/                    # Original CSV datasets
│   └── processed/              # Cleaned datasets
├── modules/
│   ├── data_loader.py          # Data loading & validation
│   ├── nlp_engine.py           # Gemini API integration
│   ├── intent_handler.py       # Intent-to-function mapping
│   ├── analytics_engine.py     # KPI & data operations
│   ├── forecasting.py          # Time-series predictions
│   ├── visualizer.py           # Chart generation
│   ├── insight_generator.py    # Text summaries & insights
│   └── report_generator.py     # PDF report export
├── prompts/
│   └── system_prompts.py       # Gemini prompt templates
├── utils/
│   ├── config.py               # App configuration
│   └── helpers.py              # Shared utilities
├── reports/                    # Generated PDF reports
└── docs/                       # Documentation
```

## 👥 Team

Built by a team of 9 members as an internship project.

## 📄 License

This project is for educational purposes.
