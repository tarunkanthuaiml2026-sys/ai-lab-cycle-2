import os
import re
import string
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def setup_datasets():
    """Creates all required CSV datasets locally if they do not exist."""
    datasets = {
        "experiment_8_student_performance.csv": """Student_ID,Name,Department,Attendance_Percentage,Assignment_Score,Exam_Score
S101,Aarav,CSE,92,88,91
S102,Diya,ECE,85,76,82
S103,Kavin,CSE,78,81,79
S104,Meera,IT,96,93,95
S105,Arjun,ECE,,72,75
S106,Nila,CSE,88,,86
S107,Rohan,IT,74,69,71
S108,Ishita,CSE,91,90,
S109,Vikram,ECE,67,73,70
S110,Ananya,IT,83,85,88
S111,Kabir,CSE,,78,80
S112,Tara,ECE,89,87,90
S113,Aditya,IT,77,74,76
S114,Sara,CSE,94,96,98
S115,Neil,ECE,81,79,84
S116,Maya,IT,86,82,85
S117,Dev,CSE,72,70,74
S118,Riya,ECE,90,91,93
S119,Om,IT,,68,72
S120,Zoya,CSE,87,84,89
S103,Kavin,CSE,78,81,79
S121,Ira,,84,80,83
""",
        "experiment_9_retail_sales.csv": """Date,Product,Category,Region,Units_Sold,Unit_Price
2026-01-05,Laptop,Electronics,North,3,52000
2026-01-12,Phone,Electronics,South,8,18000
2026-01-18,Chair,Furniture,East,6,3500
2026-01-24,Desk,Furniture,West,4,7200
2026-02-03,Laptop,Electronics,East,2,52000
2026-02-10,Phone,Electronics,West,10,18000
2026-02-16,Chair,Furniture,North,7,3500
2026-02-25,Desk,Furniture,South,5,7200
2026-03-04,Laptop,Electronics,South,4,52000
2026-03-11,Phone,Electronics,North,9,18000
2026-03-17,Chair,Furniture,West,5,3500
2026-03-26,Desk,Furniture,East,6,7200
2026-04-02,Laptop,Electronics,West,3,52000
2026-04-09,Phone,Electronics,East,11,18000
2026-04-15,Chair,Furniture,South,8,3500
2026-04-23,Desk,Furniture,North,7,7200
2026-05-06,Laptop,Electronics,North,5,52000
2026-05-13,Phone,Electronics,South,12,18000
2026-05-19,Chair,Furniture,East,9,3500
2026-05-27,Desk,Furniture,West,8,7200
2026-06-03,Laptop,Electronics,East,4,52000
2026-06-10,Phone,Electronics,West,13,18000
2026-06-18,Chair,Furniture,North,10,3500
2026-06-26,Desk,Furniture,South,9,7200
""",
        "experiment_10_prompt_evaluation.csv": """Prompt_ID,Task,Weak_Prompt,Improved_Prompt
P101,Explain photosynthesis,Explain photosynthesis.,"Act as a biology tutor. Explain photosynthesis for a Grade 8 student in 120 words. Include sunlight, chlorophyll, carbon dioxide and water. Give one example."
P102,Write a formal email,Write an email.,"Act as a communication assistant. Draft a polite email to a project manager requesting a two-day report extension. Include a subject and keep the body under 120 words."
P103,Python debugging,Fix my code.,"Act as a Python mentor. Review the supplied function, identify the IndexError, provide corrected code and explain the fix in three bullet points."
P104,Summarize a report,Summarize this.,"Act as a research assistant. Summarize the supplied report for a college student in five bullet points. Preserve key facts and do not add unsupported information."
P105,Compare algorithms,Which is better?,"Act as an algorithms instructor. Compare merge sort and bubble sort in a table using time complexity, space complexity, stability and use cases."
""",
        "experiment_11_prompting_techniques.csv": """Example_ID,Technique,Role,Task,Input_Text,Example_Input,Example_Output
T101,Zero-Shot,Sentiment analyst,Classify review sentiment,The delivery was quick and the product works perfectly.,,
T102,One-Shot,Customer feedback analyst,Classify review sentiment,The battery stopped working after one day.,The service was excellent and helpful.,Positive
T103,Few-Shot,Product review analyst,Classify review sentiment,The camera quality is poor and the screen is dull.,The package arrived early and I love the quality.,Positive
T104,Role-Based,Academic writing tutor,Rewrite the sentence professionally,"hey, send me the project file now",,
T105,Few-Shot,Support analyst,Classify ticket priority,The application is unavailable for every user.,Login is slow for one user.,Low
""",
        "experiment_12_technical_generation.csv": """Task_ID,Topic,Task_Description,Audience,Requirements
G101,CSV Data Analysis,Create a Python function to calculate the average of numeric values in a CSV column.,Beginner Python student,"Validate the column, ignore missing values and round the average to two decimals."
G102,Temperature Conversion,Generate a function to convert Celsius to Fahrenheit.,Beginner Python student,Accept numeric input and use F = (C x 9/5) + 32.
G103,Password Validation,Generate a password-strength checker.,Intermediate Python student,"Check length, uppercase, lowercase, digit and special character."
G104,List Statistics,"Generate a function to find minimum, maximum and range.",Beginner Python student,Reject an empty list and return the three statistics.
G105,Text Analysis,Generate a function to count total and unique words.,Beginner Python student,Normalize case and remove punctuation before counting.
""",
        "experiment_13_study_materials.csv": """Material_ID,Topic,Concept,Explanation,Keywords,Quiz_Question,Answer,Difficulty
S101,Python,Functions,A function is a reusable block of code that performs a specific task. It can accept parameters and return a value.,function parameter return reusable,What keyword defines a function in Python?,def,Easy
S102,Python,Lists,A list is an ordered and mutable collection that can store multiple items. List indexing starts at zero.,list mutable ordered index,Are Python lists mutable or immutable?,Mutable,Easy
S103,Artificial Intelligence,Machine Learning,Machine learning is a branch of AI in which systems learn patterns from data to make predictions or decisions.,machine learning data patterns prediction,What do machine learning systems learn from?,Data,Easy
S104,Artificial Intelligence,Neural Networks,A neural network is a computational model inspired by connected neurons. It contains layers of nodes that learn patterns through training.,neural network layers nodes training,What are the basic units in a neural network commonly called?,Nodes or neurons,Medium
S105,Data Science,Data Cleaning,"Data cleaning identifies and corrects missing, duplicate, inconsistent or inaccurate data before analysis.",data cleaning missing duplicate inconsistent,Name two problems addressed during data cleaning.,Missing values and duplicate records,Easy
S106,Data Science,Data Visualization,"Data visualization presents information graphically using charts and plots to reveal patterns, trends and comparisons.",visualization charts patterns trends,Which chart is useful for showing change over time?,Line chart,Easy
""",
    }

    for filename, content in datasets.items():
        if not os.path.exists(filename):
            with open(filename, "w", encoding="utf-8", newline="") as f:
                f.write(content.strip() + "\n")


# ==============================================================================
# EXPERIMENT 8: Data Cleaning, Exploratory Data Analysis and Visualization
# ==============================================================================
def run_ex8():
    print("\n" + "=" * 60)
    print("EX 8: DATA CLEANING, EDA & VISUALIZATION")
    print("=" * 60)

    df = pd.read_csv("experiment_8_student_performance.csv")
    print("----- ORIGINAL DATASET -----")
    print(df.head())
    print("\nDataset Shape:", df.shape)

    print("\n----- MISSING VALUES -----")
    print(df.isnull().sum())
    print("\nDuplicate Records:", df.duplicated().sum())

    # Handle missing numerical values
    num_cols = df.select_dtypes(include=np.number).columns
    for col in num_cols:
        df[col] = df[col].fillna(df[col].median())

    # Handle missing categorical values
    cat_cols = df.select_dtypes(include="object").columns
    for col in cat_cols:
        df[col] = df[col].fillna(df[col].mode()[0])

    # Remove duplicate records
    df.drop_duplicates(inplace=True)

    # Calculate average score
    df["Average_Score"] = df[["Assignment_Score", "Exam_Score"]].mean(axis=1)

    # Classify student performance
    df["Performance"] = pd.cut(
        df["Average_Score"],
        bins=[-1, 50, 70, 85, 100],
        labels=["Needs Improvement", "Average", "Good", "Excellent"],
    )

    print("\n----- CLEANED DATASET -----")
    print(df.head(10))
    print("\nCleaned Dataset Shape:", df.shape)

    print("\n----- STATISTICAL SUMMARY -----")
    print(df.describe().round(2))

    dept_analysis = df.groupby("Department")["Exam_Score"].mean()
    print("\n----- DEPARTMENT-WISE AVERAGE -----")
    print(dept_analysis.round(2))

    print("\n----- PERFORMANCE DISTRIBUTION -----")
    print(df["Performance"].value_counts())

    # Visualization 1: Department-wise average score
    plt.figure(figsize=(8, 5))
    dept_analysis.plot(kind="bar", color="skyblue", edgecolor="black")
    plt.title("Department-wise Average Exam Score")
    plt.xlabel("Department")
    plt.ylabel("Average Exam Score")
    plt.xticks(rotation=0)
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.show()

    # Visualization 2: Student performance distribution
    plt.figure(figsize=(8, 5))
    df["Performance"].value_counts().plot(kind="bar", color="orange", edgecolor="black")
    plt.title("Student Performance Distribution")
    plt.xlabel("Performance Category")
    plt.ylabel("Number of Students")
    plt.xticks(rotation=15)
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.show()

    print("\nData Cleaning and EDA Completed Successfully!")


# ==============================================================================
# EXPERIMENT 9: Data Visualization Using Matplotlib
# ==============================================================================
def run_ex9():
    print("\n" + "=" * 60)
    print("EX 9: DATA VISUALIZATION USING MATPLOTLIB")
    print("=" * 60)

    df = pd.read_csv("experiment_9_retail_sales.csv")
    df["Date"] = pd.to_datetime(df["Date"])
    df["Revenue"] = df["Units_Sold"] * df["Unit_Price"]

    print("----- RETAIL SALES DATASET -----")
    print(df.head())

    print("\nTotal Units Sold:", df["Units_Sold"].sum())
    print("Total Revenue: Rs.", df["Revenue"].sum())

    monthly_sales = df.groupby(df["Date"].dt.to_period("M"))["Revenue"].sum()
    print("\n----- MONTHLY REVENUE -----")
    print(monthly_sales)

    category_sales = df.groupby("Category")["Revenue"].sum()
    print("\n----- CATEGORY-WISE REVENUE -----")
    print(category_sales)

    region_sales = df.groupby("Region")["Revenue"].sum()
    print("\n----- REGION-WISE REVENUE -----")
    print(region_sales)

    # Visualization 1: Monthly Revenue Line Chart
    plt.figure(figsize=(9, 5))
    plt.plot(
        monthly_sales.index.astype(str),
        monthly_sales.values,
        marker="o",
        linewidth=2,
        label="Monthly Revenue",
    )
    plt.title("Monthly Retail Revenue")
    plt.xlabel("Month")
    plt.ylabel("Revenue (Rs.)")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.show()

    # Visualization 2: Category-wise Bar Chart
    plt.figure(figsize=(8, 5))
    category_sales.plot(kind="bar", color="lightgreen", edgecolor="black")
    plt.title("Category-wise Revenue Comparison")
    plt.xlabel("Product Category")
    plt.ylabel("Revenue (Rs.)")
    plt.xticks(rotation=0)
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.show()

    # Visualization 3: Region-wise Pie Chart
    plt.figure(figsize=(7, 7))
    plt.pie(
        region_sales,
        labels=region_sales.index,
        autopct="%1.1f%%",
        startangle=90,
    )
    plt.title("Region-wise Revenue Distribution")
    plt.tight_layout()
    plt.show()

    # Visualization 4: Units Sold vs Revenue
    plt.figure(figsize=(8, 5))
    plt.scatter(df["Units_Sold"], df["Revenue"], color="red", marker="o")
    plt.title("Units Sold vs Revenue")
    plt.xlabel("Units Sold")
    plt.ylabel("Revenue (Rs.)")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.show()

    print("\nData Visualization Completed Successfully!")


# ==============================================================================
# EXPERIMENT 10: Prompt Design and Evaluation Using Generative AI
# ==============================================================================
def run_ex10():
    print("\n" + "=" * 60)
    print("EX 10: PROMPT DESIGN AND EVALUATION")
    print("=" * 60)

    df = pd.read_csv("experiment_10_prompt_evaluation.csv")

    def evaluate_prompt(prompt):
        text = prompt.lower()
        features = {
            "Role": any(word in text for word in ["act as", "you are"]),
            "Context": any(
                word in text for word in ["student", "manager", "college", "project", "report"]
            ),
            "Clear task": any(
                word in text for word in ["explain", "draft", "review", "summarize", "compare"]
            ),
            "Constraints": any(
                word in text for word in ["words", "bullet", "under", "three", "five"]
            ),
            "Output format": any(
                word in text for word in ["table", "subject", "bullet points", "example"]
            ),
        }
        return sum(features.values()), features

    print("----- PROMPT DESIGN AND EVALUATION -----")
    for _, row in df.iterrows():
        weak_score, _ = evaluate_prompt(row["Weak_Prompt"])
        improved_score, features = evaluate_prompt(row["Improved_Prompt"])
        print(f"\nPrompt ID: {row['Prompt_ID']} | Task: {row['Task']}")
        print(f"Weak prompt score    : {weak_score}/5")
        print(f"Improved prompt score: {improved_score}/5")
        print("Satisfied features   :", ", ".join(k for k, v in features.items() if v))
        print("Evaluation           :", "Improved" if improved_score > weak_score else "Needs review")

    df["Weak_Score"] = df["Weak_Prompt"].apply(lambda x: evaluate_prompt(x)[0])
    df["Improved_Score"] = df["Improved_Prompt"].apply(lambda x: evaluate_prompt(x)[0])

    print("\n----- OVERALL SUMMARY -----")
    print("Average weak prompt score    :", round(df["Weak_Score"].mean(), 2))
    print("Average improved prompt score:", round(df["Improved_Score"].mean(), 2))
    print("\nEvaluation completed. This is a rule-based prompt-quality checklist, not an LLM evaluation.")


# ==============================================================================
# EXPERIMENT 11: Role-Based, Zero-Shot, One-Shot and Few-Shot Prompting
# ==============================================================================
def run_ex11():
    print("\n" + "=" * 60)
    print("EX 11: ROLE-BASED, ZERO-SHOT, ONE-SHOT & FEW-SHOT PROMPTING")
    print("=" * 60)

    df = pd.read_csv("experiment_11_prompting_techniques.csv")

    positive_words = {"quick", "perfectly", "excellent", "helpful", "love", "early", "great"}
    negative_words = {"poor", "dull", "stopped", "broken", "slow", "unavailable"}

    def sentiment_demo(sentence):
        words = set(sentence.lower().replace(".", "").replace(",", "").split())
        score = len(words & positive_words) - len(words & negative_words)
        return "Positive" if score > 0 else "Negative" if score < 0 else "Neutral"

    def construct_prompt(row):
        role = row["Role"]
        task = row["Task"]
        text = row["Input_Text"]
        technique = row["Technique"]

        if technique == "Zero-Shot":
            return f"You are a {role}. {task}: {text}"
        if technique == "One-Shot":
            return (
                f"You are a {role}. {task}. Example: "
                f"{row['Example_Input']} -> {row['Example_Output']}. New input: {text}"
            )
        if technique == "Few-Shot":
            return (
                f"You are a {role}. {task}. Examples: "
                "Excellent service -> Positive; Poor quality -> Negative. "
                f"New input: {text}"
            )
        return f"Act as a {role}. {task}: {text}"

    print("----- ROLE-BASED AND EXAMPLE-BASED PROMPTING -----")
    for _, row in df.iterrows():
        prompt = construct_prompt(row)
        if "sentiment" in row["Task"].lower():
            response = sentiment_demo(row["Input_Text"])
        elif "priority" in row["Task"].lower():
            response = "High" if "unavailable" in row["Input_Text"].lower() else "Low"
        else:
            response = "Suggested rewrite: Please share the project file at your earliest convenience."

        print(f"\nID: {row['Example_ID']} | Technique: {row['Technique']}")
        print("Constructed prompt:", prompt)
        print("Python demonstration response:", response)

    print("\nDemonstration completed. Responses use predefined Python rules, not a live LLM.")


# ==============================================================================
# EXPERIMENT 12: AI-Assisted Generation of Technical Content and Python Programs
# ==============================================================================
def run_ex12():
    print("\n" + "=" * 60)
    print("EX 12: TECHNICAL CONTENT & PYTHON PROGRAM GENERATION")
    print("=" * 60)

    df = pd.read_csv("experiment_12_technical_generation.csv")

    templates = {
        "CSV Data Analysis": '''import pandas as pd

def column_average(filename, column):
    data = pd.read_csv(filename)
    if column not in data.columns:
        raise ValueError("Column not found")
    values = pd.to_numeric(data[column], errors="coerce").dropna()
    if values.empty:
        raise ValueError("No numeric values available")
    return round(values.mean(), 2)
''',
        "Temperature Conversion": '''def celsius_to_fahrenheit(celsius):
    if not isinstance(celsius, (int, float)):
        raise TypeError("Temperature must be numeric")
    return round((celsius * 9 / 5) + 32, 2)
''',
        "Password Validation": '''def check_password(password):
    checks = {
        "Minimum 8 characters": len(password) >= 8,
        "Uppercase": any(c.isupper() for c in password),
        "Lowercase": any(c.islower() for c in password),
        "Digit": any(c.isdigit() for c in password),
        "Special character": any(not c.isalnum() for c in password),
    }
    for rule, passed in checks.items():
        print(rule, ":", "Pass" if passed else "Fail")
    return all(checks.values())
''',
        "List Statistics": '''def list_statistics(values):
    if not values:
        raise ValueError("List cannot be empty")
    return {
        "minimum": min(values),
        "maximum": max(values),
        "range": max(values) - min(values),
    }
''',
        "Text Analysis": '''import string

def analyze_text(text):
    cleaned = text.translate(str.maketrans("", "", string.punctuation)).lower()
    words = cleaned.split()
    return {"total_words": len(words), "unique_words": len(set(words))}
''',
    }

    output_dir = Path("generated_technical_content")
    output_dir.mkdir(exist_ok=True)

    print("----- TECHNICAL CONTENT AND PYTHON PROGRAM GENERATION -----")
    for _, row in df.iterrows():
        topic = row["Topic"]
        code = templates[topic]
        explanation = (
            f"Topic: {topic}\n"
            f"Audience: {row['Audience']}\n"
            f"Task: {row['Task_Description']}\n"
            f"Requirements: {row['Requirements']}\n"
            "Approach: validate inputs, process data and present a clear result."
        )
        file_stem = row["Task_ID"].lower()
        (output_dir / f"{file_stem}.py").write_text(code, encoding="utf-8")
        (output_dir / f"{file_stem}_notes.txt").write_text(explanation, encoding="utf-8")

        print(f"\nTask ID: {row['Task_ID']} | Topic: {topic}")
        print(explanation)
        print("Saved:", f"{file_stem}.py", "and", f"{file_stem}_notes.txt")

    print("\nTotal generated files:", len(list(output_dir.iterdir())))
    print("Template-based generation completed. No external LLM is called.")


# ==============================================================================
# EXPERIMENT 13: AI-Powered Study Assistant Using Python and Generative AI
# ==============================================================================
def run_ex13():
    print("\n" + "=" * 60)
    print("EX 13: AI-POWERED STUDY ASSISTANT")
    print("=" * 60)

    df = pd.read_csv("experiment_13_study_materials.csv")

    def tokenize(text):
        return set(re.findall(r"[a-zA-Z]+", str(text).lower()))

    def retrieve_material(query):
        query_words = tokenize(query)
        scores = []
        for _, row in df.iterrows():
            searchable_text = " ".join(
                [row["Topic"], row["Concept"], row["Explanation"], row["Keywords"]]
            )
            material_words = tokenize(searchable_text)
            score = len(query_words & material_words) / max(len(query_words), 1)
            scores.append(score)

        best_index = scores.index(max(scores))
        if scores[best_index] == 0:
            return None, 0
        return df.iloc[best_index], scores[best_index]

    def study_assistant(query):
        material, score = retrieve_material(query)
        if material is None:
            print("No matching material found. Try another concept.")
            return

        print("\n----- PERSONALIZED STUDY NOTES -----")
        print("Topic      :", material["Topic"])
        print("Concept    :", material["Concept"])
        print("Explanation:", material["Explanation"])
        print("Match score:", round(score * 100, 2), "%")

        print("\n----- QUICK QUIZ -----")
        print("Question   :", material["Quiz_Question"])
        print("Difficulty :", material["Difficulty"])
        print("Answer     :", material["Answer"])

    print("Available concepts:", ", ".join(df["Concept"].tolist()))
    query = input("\nEnter a topic or concept to study (e.g., Neural Networks): ")
    study_assistant(query)
    print("\nStudy assistant session completed.")
    print("Note: Matching is based on CSV keyword overlap, not a live Generative AI model.")


# ==============================================================================
# MAIN MENU
# ==============================================================================
if __name__ == "__main__":
    setup_datasets()

    experiments = {
        "8": ("Data Cleaning, EDA & Visualization", run_ex8),
        "9": ("Data Visualization Using Matplotlib", run_ex9),
        "10": ("Prompt Design and Evaluation", run_ex10),
        "11": ("Role-Based, Zero-Shot, One-Shot & Few-Shot Prompting", run_ex11),
        "12": ("Technical Content & Python Program Generation", run_ex12),
        "13": ("AI-Powered Study Assistant", run_ex13),
    }

    while True:
        print("\n" + "*" * 60)
        print("EXPLORATION IN ARTIFICIAL INTELLIGENCE (EXPERIMENTS 8 - 13)")
        print("*" * 60)
        for key, (title, _) in experiments.items():
            print(f"  {key}. {title}")
        print("  A. Run All Experiments (8 to 13)")
        print("  Q. Quit")

        choice = input("\nSelect an option (8-13, A, or Q): ").strip().upper()

        if choice == "Q":
            print("Exiting...")
            break
        elif choice == "A":
            for _, (_, func) in experiments.items():
                func()
        elif choice in experiments:
            experiments[choice][1]()
        else:
            print("Invalid choice. Please enter 8, 9, 10, 11, 12, 13, A, or Q.")