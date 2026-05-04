# 📚🍜 UBC Food Psychology Study
UBC Food Psychology Study @ UBC is a data analysis project that explores how budget constraints, mood, academic stress, and exam periods influence food choices among UBC students. Using student survey data and UBC-specific food options, this project examines whether food decisions are driven more by psychological factors or financial limitations, and how stress affects delivery and comfort food consumption.

The project applies data cleaning, feature engineering, exploratory data analysis, and statistical testing to uncover behavioral patterns in student eating habits. The findings highlight the dominant role of budget constraints in shaping food choices, while showing that academic stress significantly increases delivery food consumption, especially during high-pressure periods.

## Key Features
- Cleaned and processed student food behavior and spending data
- Engineered indicators for mood, budget constraints, stress, and comfort food behavior
- Built a composite academic stress score using exams, study hours, and emotional state
- Visualized trends using exploratory data analysis techniques
- Applied statistical tests (t-tests, ANOVA, chi-square) to validate insights

## Research Questions
1. What influences food choices more: mood or money?
2. How do academic stress and exam periods affect delivery and comfort food consumption?

## Key Findings
- **Budget > Mood**: Budget constraints significantly influence both spending 
  and food type choices (p < 0.05), while mood does not (p = 0.31)
- **Stress → Delivery**: Academic stress significantly increases delivery 
  food orders (p = 0.001)
- **Comfort Food**: Neither stress nor exam periods significantly predict 
  comfort food consumption (p > 0.05)

## Dataset
- `students_dataset.csv` — student survey data including mood, budget, 
  study hours, exam period, and food ordering behaviour
- `food_options_dataset.csv` — UBC-specific food options and pricing data

## Tech Stack
Python, pandas, NumPy, matplotlib, seaborn, SciPy

