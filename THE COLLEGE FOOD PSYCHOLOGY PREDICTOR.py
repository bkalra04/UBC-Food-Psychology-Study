# THE COLLEGE FOOD PSYCHOLOGY PREDICTOR 

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy import stats
import warnings

import os
SAVE_PLOTS = True   # write charts to /figures instead of opening windows
if SAVE_PLOTS:
    import matplotlib
    matplotlib.use("Agg")
    os.makedirs("figures", exist_ok=True)

pd.set_option("display.max_columns", None)  # Show all columns when printing datasets
pd.set_option("display.width", 1000)         # Make the display wider for better readability

print("=== THE COLLEGE FOOD PSYCHOLOGY PREDICTOR ===")
print("STEP 1: Loading and Exploring Datasets\n")

# STEP 1. LOAD DATASETS
try:
    students_df = pd.read_csv("students_dataset.csv")
    food_options_df = pd.read_csv("food_options_dataset.csv")

    print("✅ All datasets loaded successfully")
    print(f"Students dataset: {students_df.shape}")   # Shows (rows, columns)
    print(f"Food Options dataset: {food_options_df.shape}")

except FileNotFoundError as e:
    # If any CSV file is missing, this error message appears.
    print(f"❌ File not found: {e}")
    print("Please ensure your CSV files are in the correct directory")

# EXPLORING THE DATASETS

print("\n" + "="*20)
print("DATA STRUCTURES")
print("="*20)

print("\n🎓 STUDENTS DATASET")
print(students_df.head())
print(f"\nColumns: {list(students_df.columns)}")
print(f"Data Types:\n{students_df.dtypes}")

print("\n" + "="*30)
print("\n🍜 FOOD OPTIONS DATASET")
print(f"\nColumns: {list(food_options_df.columns)}")
print(f"Data Types:\n{food_options_df.dtypes}")

print("\n" + "="*10)
print("NEXT STEPS")
print("="*10)
print("1. ✅ Data loaded and structure verified")
print("2. 🔄 Next: Data merging and derived metrics creation")
print("3. 📈 Then: Exploratory data analysis")
print("4. 🎯 Finally: Core research questions analysis")

# STEP 2. DATA PREPARATION 
# This step cleans data and creates variables needed for our research questions:
# Q1: Which factor is stronger in deciding food choices - MOOD or MONEY?
# Q2: How do exams and academic stress affect food choices and comfort food consumption?

print("\n" + "="*20)
print("DATA PREPARATION")
print("="*20)

# Data Cleaning and Type Conversion
print("\n🧹 CLEANING DATA")

# 1. Convert budget ranges to numeric (took midpoint of the ranges)
def convert_budget_to_numeric(budget_str):
    """
    Convert the range prices into numeric values (done by taking its midpoint)
    """
    budget_str = str(budget_str).strip()
    if pd.isna(budget_str) or budget_str == "":
        return None
    try:
        # Handle single numbers
        if "-" not in str(budget_str):
            return float(budget_str)
        
        # Handle ranges
        range_parts = str(budget_str).split("-")
        if len(range_parts) == 2:
            return (float(range_parts[0]) + float(range_parts[1])) / 2
    except (ValueError, TypeError):
        return None
    return None

students_df = students_df.loc[:, ~students_df.columns.str.startswith("Unnamed")]
print(f"✅ Dropped empty columns — {students_df.shape[1]} remain")

students_df["budget_numeric"] = students_df["budget ($)"].apply(convert_budget_to_numeric)

# 2. Convert delivery time ranges into numerics (took midpoint)
def convert_delivery_time_to_numeric(time_str):
    """
    Convert the delivery time ranges to numeric (done by taking midpoint)
    """
    time_str = str(time_str).strip()
    if pd.isna(time_str) or time_str == "":
        return None
    try:
        # Handle single numbers
        if "-" not in str(time_str):
            return float(time_str)
        
        # Handle ranges
        range_parts = str(time_str).split("-")
        if len(range_parts) == 2:
            return (float(range_parts[0]) + float(range_parts[1])) / 2
    
    except (ValueError, TypeError):
        return None
    return None

students_df["delivery_time_numeric"] = students_df["delivery_time"].apply(convert_delivery_time_to_numeric)

# 3. Convert study hours ranges into numeric

def convert_study_hours_to_numeric(hours_str):
    """
    Convert study hour ranges into numeric (taking midpoint)
    """
    hours_str = str(hours_str).strip()
    if pd.isna(hours_str) or hours_str == "":
        return None
    try:
        # Handle singe numbers
        if "-" not in str(hours_str):
            return float(hours_str)

        # Handle ranges
        range_parts = str(hours_str).split("-")
        if len(range_parts) == 2:
            return (float(range_parts[0]) + float(range_parts[1])) / 2

    except (ValueError, TypeError):
        return None
    return None

students_df["study_hours_numeric"] = students_df["study_hours_today"].apply(convert_study_hours_to_numeric)

print("✅ Converted budget, delivery time, and study hours into numeric values.")

for col in ["budget_numeric", "delivery_time_numeric", "study_hours_numeric"]:
    failed = students_df[col].isna().sum()
    print(f"   {col}: {failed} values failed to parse ({failed/len(students_df)*100:.1f}%)")

# RESEARCH QUESTION 1: WHAT INFLUENCES FOOD CHOICES - MOOD VS MONEY?

def budget_upper_bound(budget_str):
    """The top of a student's stated range — the real overspending threshold."""
    budget_str = str(budget_str).strip()
    if "-" not in budget_str:
        try:
            return float(budget_str)
        except (ValueError, TypeError):
            return None
    parts = budget_str.split("-")
    if len(parts) == 2:
        try:
            return float(parts[1])
        except (ValueError, TypeError):
            return None
    return None

students_df["budget_max"] = students_df["budget ($)"].apply(budget_upper_bound)

# Create spending behaviour indicators
students_df["overspent"] = students_df["actual_amount_spent"] > students_df["budget_max"]
students_df["spending_ratio"] = students_df["actual_amount_spent"] / students_df["budget_numeric"]
students_df["budget_difference"] = students_df["actual_amount_spent"] - students_df["budget_max"]

# Create mood categories for analysis
mood_categories = {
    "positive" : ["Happy", "Excited", "Energetic"],
    "negative" : ["Stressed", "Sad", "Tired", "Anxious", "Homesick", "Lazy"],
    "neutral" : ["Neutral", "Calm", "Focused"]
}

def categorize_mood(mood):
    """
    Categorize mood into positive, negative, and neutral.
    """
    if pd.isna(mood):
        return "Unknown"
    mood = str(mood).strip()
    for category, moods in mood_categories.items():
        if mood in moods:
            return category
    return "Other"

students_df["mood_category"] = students_df["mood"].apply(categorize_mood)

# Create financial constraint levels
def budget_constraint_level(budget):
    """
    Categorize budget levels into low, medium, and high
    """
    if pd.isna(budget):
        return "Unknown"
    if budget <= 15:
        return "low budget"
    elif budget <= 22:
        return "medium budget"
    else:
        return "high budget"
    
students_df["budget_constraint"] = students_df["budget_numeric"].apply(budget_constraint_level)

print("✅ Created Mood VS Money variables")
print(" - Spending Behaviour Indicators")
print(" - Mood Categories (positive/negative/neutral)")
print(" - Budget Constraint levels (low/medium/high)")

# RESEARCH QUESTION 2: EXAMS + ACADEMIC STRESS AND DELIVERY + COMFORT FOOD CONSUMPTION

# Create comprehensive stress indicators
def calculate_stress_score(row):
    """
    Calculate overall stress score based on multiple factors.
    """
    stress_score = 0

    # Exam period contribution
    if pd.notna(row["exam_period"]):
        if "Exam Week" in str(row["exam_period"]):
            stress_score += 3
        elif "Pre Exam" in  str(row["exam_period"]):
            stress_score += 2

    # Academic Stress contribution
    if pd.notna(row["academic_stress_level"]):
        stress_level = str(row["academic_stress_level"])
        if stress_level == "High":
            stress_score += 3
        elif stress_level == "Medium":
            stress_score += 2
        elif stress_level == "Low":
            stress_score += 1
    

    # Study hours contribution (more hours = more stress)
    if pd.notna(row["study_hours_numeric"]):
        if row["study_hours_numeric"] >= 10:
            stress_score += 2
        elif row["study_hours_numeric"] >= 6:
            stress_score += 1


    # Mood Contribution
    if row["mood_category"] == "negative":
        stress_score += 1
    
    return stress_score

students_df["overall_stress_score"] = students_df.apply(calculate_stress_score, axis=1)

# Categorize stress levels
def stress_category(score):
    """
    Convert stress score into categories.
    """
    if score >= 7:
        return "Very High Stress"
    elif score >= 5:
        return "High Stress"
    elif score >= 3:
        return "Moderate Stress"
    elif score >= 1:
        return "Low Stress"
    else:
        return "No Stress"
    
students_df["stress_category"] = students_df["overall_stress_score"].apply(stress_category)

# Create delivery behaviour indicators
students_df["ordered_delivery"] = (students_df["ordered_food"] == "Yes") & (students_df["food_type"] == "Delivery")
students_df["chose_microwave"] = (students_df["food_type"] == "Microwave")

# Create comfort food indicators (based on low healthiness and high spending)
def is_comfort_food_behaviour(row):
    """
    Comfort eating, defined only by what was eaten and what was spent.

    An earlier version also counted "ordered delivery while stressed",
    which meant stress was baked into the outcome we then tested stress
    against. Both indicators here are independent of stress.
    """
    comfort_indicators = 0

    # Low healthiness score
    if pd.notna(row["healthiness_level"]) and row["healthiness_level"] <= 2:
        comfort_indicators += 1

    # Spending past their own stated maximum — impulsive rather than planned
    if row["overspent"]:
        comfort_indicators += 1

    return comfort_indicators >= 2


print("Current columns in students_df:")
print(students_df.columns)
students_df["comfort_food_behaviour"] = students_df.apply(is_comfort_food_behaviour, axis=1)

print(students_df["comfort_food_behaviour"].value_counts())
print(students_df["overspent"].value_counts())

print("✅ Created academic stress variables:")
print(" - Overall Stress Score (0-10 scale)")
print(" - Stress Categories (No to very High)")
print(" - Delivery Behaviour Indicators")
print(" - Comfort Food Behaviour Indicators")

# Data Quality Check after Processing

# Check for missing values in key variables
key_variables = ["budget_numeric", "mood_category", "stress_category", "overall_stress_score"]
print("\nMissing Values in Key Variables:")
for var in key_variables:
    missing_count = students_df[var].isnull().sum()
    missing_pct = (missing_count / len(students_df)) * 100
    print(f" {var}: {missing_count} ({missing_pct:.1f}%)")

# Show distribution of key categorical variables
print("\n📊 KEY VARIABLE DISTRIBUTIONS:")
print(f"\n🎭 Mood Categories:")
print(students_df["mood_category"].value_counts())

print(f"\n💰 Budget Constraint Levels:")
print(students_df["budget_constraint"].value_counts())

print(f"\n😰 Stress Categories:")
print(students_df["stress_category"].value_counts())

print(f"\n🚚 Delivery Behaviour:")
delivery_count = students_df["ordered_delivery"].sum()
total_orders = students_df["ordered_food"].value_counts().get("Yes", 0)
print(f"Delivery Orders: {delivery_count}")
print(f"Total Food Orders: {total_orders}")
if total_orders > 0:
    print(f"Delivery Rate: {(delivery_count / total_orders) * 100:.1f}%")

# Summary
print("\n" + "="*40)
print("SUMMARY OF PREPARED VARIABLES")
print("="*40)

print(f"\n📊 Dataset Shape After Processing: {students_df.shape}")
print(f"Original columns: 17")
print(f"New columns added: {students_df.shape[1] - 17}")

print(f"\n🎯 RESEARCH QUESTION 1 VARIABLES (Mood vs Money):")
q1_vars = ['mood_category', 'budget_constraint', 'overspent', 'spending_ratio', 'budget_difference']
for var in q1_vars:
    print(f"  ✅ {var}")

print(f"\n🎯 RESEARCH QUESTION 2 VARIABLES (Academic Stress):")
q2_vars = ['overall_stress_score', 'stress_category', 'ordered_delivery', 'comfort_food_behaviour']
for var in q2_vars:
    print(f"  ✅ {var}")

print(f"\n📋 SAMPLE OF PROCESSED DATA:")
display_cols = ['student_id', 'mood_category', 'budget_constraint', 'stress_category', 
                'ordered_delivery', 'comfort_food_behaviour', 'overspent']
print(students_df[display_cols].head(10))

print("\n" + "="*10)
print("NEXT STEPS")
print("="*10)
print("✅ Step 2 Complete: Data prepared for analysis")
print("🔄 Next: Step 3 - Exploratory Data Analysis")
print("📊 Then: Step 4 - Core Research Questions Analysis")
print("🎯 Finally: Statistical testing and insights")

# STEP 3. EXPLANATORY DATA ANALYSIS (EDA)

# 3.1 Actual Amount Spent by Mood Category
plt.figure(figsize=(6,4))
sns.boxplot(x="mood_category", y="actual_amount_spent", data=students_df)
plt.title("Actual Amount Spent per Mood Category")
plt.savefig("figures/3-1-spending-by-mood.png", dpi=150, bbox_inches="tight")
plt.close()

# INTERPREATION:
# Shows if spending changes alot between moods. 
# Since the difference is not big, this suggests "Mood" does not have a stronger impact on food spending decisions.

# 3.2 Actual Amount Spent by Budget Constraint
plt.figure(figsize=(6,4))
sns.boxplot(x="budget_constraint", y="actual_amount_spent", data=students_df)
plt.title("Actual Amount Spent per Budget Constraint")
plt.savefig("figures/3-2-spending-by-budget.png", dpi=150, bbox_inches="tight")
plt.close()

# INTERPREATION:
# Shows if spending changes alot between budget levels. 
# Since the difference is big, this suggests "Money" has a stronger impact on food spending decisions.

# 3.3 Food Type by Mood Category
plt.figure(figsize=(6,4))
sns.countplot(x="mood_category", hue="food_type", data=students_df)
plt.title("Food Type by Mood Category")
plt.savefig("figures/3-3-food-type-by-mood.png", dpi=150, bbox_inches="tight")
plt.close()

# INTERPREATION:
# Shows if mood changes what type of food people choose.

# 3.4 Food Type by Budget Constraint
plt.figure(figsize=(6,4))
sns.countplot(x="budget_constraint", hue="food_type", data=students_df)
plt.title("Food Type by Budget Constraint")
plt.savefig("figures/3-4-food-type-by-budget.png", dpi=150, bbox_inches="tight")
plt.close()

# INTERPREATION:
# Shows if budget levels changes what type of food people choose.

# 3.5 Comfort Food Behaviour by Stress Category
plt.figure(figsize=(6,4))
sns.barplot(x="stress_category", y="comfort_food_behaviour", data=students_df)
plt.title("Comfort Food Behaviour by Stress Category")
plt.savefig("figures/3-5-comfort-food-by-stress.png", dpi=150, bbox_inches="tight")
plt.close()

# INTERPREATION:
# Shows if stress makes comfort food the more likely choice.

# 3.6 Delivery Orders by Stress Category
plt.figure(figsize=(6,4))
sns.barplot(x="stress_category", y="ordered_delivery", data=students_df)
plt.title("Delivery Orders by Stress Category")
plt.savefig("figures/3-6-delivery-orders-by-stress.png", dpi=150, bbox_inches="tight")
plt.close()

# INTERPREATION:
# Shows if stress makes delivery food the more likely choice.

# 3.7 Comfort Food Behaviour by Exam Period
plt.figure(figsize=(6,4))
sns.barplot(x="exam_period", y="comfort_food_behaviour", data=students_df)
plt.title("Comfort Food Behaviour by Exam Period")
plt.savefig("figures/3-7-comfort-food-by-exam.png", dpi=150, bbox_inches="tight")
plt.close()

# INTERPREATION:
# Shows if comfort food is more common during exams.

print("\n" + "="*10)
print("NEXT STEPS")
print("="*10)
print("✅ Step 2 Complete: Data prepared for analysis")
print("✅ Step 3 Complete: Exploratory Data Analysis")
print("🔄 Next: Step 4 - Core Research Questions Analysis")
print("🎯 Finally: Statistical testing and insights")

# STEP 4: CORE RESEARCH QUESTIONS ANALYSIS

# QUESTION 1 - WHAT INFLUENCES FOOD CHOICES MORE: MOOD OR MONEY?
# A) Does "mood" influence actual_amount_spent

from scipy.stats import ttest_ind

# Subset by Mood
spending_positive = students_df[students_df["mood_category"] == "positive"]["actual_amount_spent"]
spending_negative = students_df[students_df["mood_category"] == "negative"]["actual_amount_spent"]

# Run independant t-test
t_mood, p_mood = ttest_ind(spending_positive, spending_negative, equal_var=False)
print(f"T-Statistic: {t_mood}, p-value: {p_mood}")

# B) Does "budget" influence actual_amount_spent

from scipy.stats import f_oneway

spending_low = students_df[students_df["budget_constraint"] == "low budget"]["actual_amount_spent"]
spending_medium = students_df[students_df["budget_constraint"] == "medium budget"]["actual_amount_spent"]
spending_high = students_df[students_df["budget_constraint"] == "high budget"]["actual_amount_spent"]

f_budget, p_budget = f_oneway(spending_low, spending_medium, spending_high)
print(f"F-Statistic: {f_budget}, p-value: {p_budget}")

# C) Does "mood" influence food_type

from scipy.stats import chi2_contingency

# Create contingency table
ct = pd.crosstab(students_df["mood_category"], students_df["food_type"])

# Run chi-square test
chi2_mood_food, p_mood_food, dof, expected = chi2_contingency(ct)
print(f"Chi2: {chi2_mood_food}, p-value: {p_mood_food}")

# D) Does "budget" influence food_type

# Create contingency table
ct = pd.crosstab(students_df["budget_constraint"], students_df["food_type"])

# Run chi-square test
chi2_budget_food, p_budget_food, dof, expected = chi2_contingency(ct)
print(f"Chi2: {chi2_budget_food}, p-value: {p_budget_food}")

# QUESTION 2 - DO STRESS AND EXAM PERIODS LEAD TO MORE DELIVERY AND COMFORT FOOD CONSUMPTION?
# A) Does stress affect "comfort_food_behaviour".

ct = pd.crosstab(students_df["stress_category"], students_df["comfort_food_behaviour"])

chi2_stress_comfort, p_stress_comfort, dof, expected = chi2_contingency(ct)
print(f"Chi2: {chi2_stress_comfort}, p-value: {p_stress_comfort}")

# B) Does "exam_period" affect comfort_food_behaviour.

ct = pd.crosstab(students_df["exam_period"], students_df["comfort_food_behaviour"])

chi2_exam_comfort, p_exam_comfort, dof, expected = chi2_contingency(ct)
print(f"Chi2: {chi2_exam_comfort}, p-value: {p_exam_comfort}")

# C) Does stress affect "ordered_delivery".

ct = pd.crosstab(students_df["stress_category"], students_df["ordered_delivery"])

chi2_stress_delivery, p_stress_delivery, dof, expected = chi2_contingency(ct)
print(f"Chi2: {chi2_stress_delivery}, p-value: {p_stress_delivery}")


print("\n" + "="*10)
print("NEXT STEPS")
print("="*10)
print("✅ Step 2 Complete: Data prepared for analysis")
print("✅ Step 3 Complete: Exploratory Data Analysis")
print("✅ Step 4 Complete: Core Research Questions Analysis")
print("🎯 Finally: Statistical testing and insights")

# STEP 5: PRESENTING STATISTICAL TESTING AND INSIGHTS

print("\n" + "="*40)
print("🎯 STATISTICAL TESTING INSIGHTS:")
print("="*40)
print(f"""
1. Spending by mood: t = {t_mood:.2f}, p = {p_mood:.3f}
   {'Significant' if p_mood < 0.05 else 'No significant difference'} — mood does not drive how much students spend.

2. Spending by budget: F = {f_budget:.2f}, p = {p_budget:.2e}
   {'Significant' if p_budget < 0.05 else 'No significant difference'} — budget is the dominant factor.

3. Food type by mood: chi2 = {chi2_mood_food:.2f}, p = {p_mood_food:.3f}
   Borderline; suggestive but not significant at 0.05.

4. Food type by budget: chi2 = {chi2_budget_food:.2f}, p = {p_budget_food:.5f}
   Significant — budget shapes what students eat, not just how much they spend.

5. Comfort food by stress: chi2 = {chi2_stress_comfort:.2f}, p = {p_stress_comfort:.3f}
   UNDERPOWERED — only 4 of 35 students met the comfort-eating definition.
   Expected cell counts fall below 5, so chi-square assumptions are not met.
   Treat as inconclusive rather than as a null result.

6. Comfort food by exam period: chi2 = {chi2_exam_comfort:.2f}, p = {p_exam_comfort:.3f}
   UNDERPOWERED for the same reason.

7. Food ordering by stress: chi2 = {chi2_stress_delivery:.2f}, p = {p_stress_delivery:.5f}
   Significant. Note: every student who ordered food chose delivery, so this
   measures ordering versus microwaving, not delivery versus other ordering.
""")

print("\n" + "="*20)
print("🏁 OVERALL SUMMARY")
print("="*20)
print(f"""
Across 35 students, budget was the dominant factor in both how much was spent
(F = {f_budget:.1f}, p < 0.001) and what was eaten (chi2 = {chi2_budget_food:.1f}, p < 0.001).
Mood showed no significant effect on spending and only a borderline one on food
type (p = {p_mood_food:.3f}).

Stress predicted whether students ordered food at all rather than microwaving
something (p < 0.001). Its relationship to comfort eating could not be tested
reliably: only 4 of 35 students met the definition, leaving expected cell counts
below the threshold chi-square requires.

This is a pilot study. The sample is small and self-reported, and the comfort
food result in particular should be treated as a question for a larger sample
rather than an answer.
""")

