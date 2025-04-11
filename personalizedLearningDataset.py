#Personalized Learning Dataset

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_ind, chi2_contingency
import scipy.stats as stats
import dash
from dash import dcc, html
import plotly.express as px

#Loading the dataset

df = pd.read_csv(r"C:\Users\muska\OneDrive\Desktop\personalized_learning_dataset (1).csv")


#OBJECTIVE -1: Analyze and Preprocess data by performing Exploratory Data Analysis

#Displaying basic info
print(df.head())
print(df.info())

missing_values = df.isnull().sum()
print("Missing Values:\n", missing_values)

duplicate_rows = df.duplicated().sum()
print("Duplicate Rows:", duplicate_rows)

#Summary statistics for numerical values
numerical_summary = df.describe()
print("Numerical Summary: \n", numerical_summary)

# Summary for categorical columns
categorical_summary = df.describe(include=["object"])
print("Categorical Summary:\n", categorical_summary)

#Plot heatmap of correlations
plt.figure(figsize=(10, 6))

correlation_matrix = df.select_dtypes(include=['float64', 'int64']).corr()
sns.heatmap(correlation_matrix, annot=True)
plt.title("Correlation Matrix of Numerical Features")
plt.show()


#OBJECTIVE -2: Develop a Personalized Engagement Score
df["Engagement_Score"] = (0.4 * df["Time_Spent_on_Videos"] +
                          0.3 * df["Forum_Participation"] +
                          0.3 * df["Assignment_Completion_Rate"])
print("Engagement Score: \n", df[["Student_ID", "Engagement_Score"]].head())
#Scatter plot
sns.scatterplot(x="Engagement_Score", y="Final_Exam_Score", data=df, hue="Dropout_Likelihood", palette="coolwarm")
plt.title("Engagement Score vs Final Exam Score")
plt.xlabel("Engagement Score")
plt.ylabel("Final Exam Score")
plt.grid(True)
plt.tight_layout()
plt.show()

# Plot histograms of numerical features
sns.set(style="whitegrid")

numerical_columns = [
    'Age', 'Time_Spent_on_Videos', 'Quiz_Attempts', 'Quiz_Scores',
    'Forum_Participation', 'Assignment_Completion_Rate',
    'Final_Exam_Score', 'Feedback_Score'
]

df[numerical_columns].hist(figsize=(16, 10), bins=20, edgecolor='black')
plt.suptitle("Histograms of Numerical Features", fontsize=16, fontweight='bold')
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()

####################################################################################

#OBJECTIVE -3: Analyze the Impact of Learning Styles on Academic Success
#Box plot
plt.figure(figsize=(10, 6))
sns.set_style("whitegrid")
sns.boxplot(x="Learning_Style", y="Final_Exam_Score", data=df, hue="Learning_Style", palette="Set2", legend=False)
plt.title("Box Plot - Final Exam Scores by Learning Style", fontsize=16)
plt.xlabel("Learning Style", fontsize=12)
plt.ylabel("Final Exam Score", fontsize=12)

plt.tight_layout()
plt.show()

#Bar plot
learning_style_avg = df.groupby("Learning_Style")["Final_Exam_Score"].mean().reset_index()

plt.figure(figsize=(8, 5))
sns.barplot(x="Learning_Style", y="Final_Exam_Score", data=learning_style_avg, hue="Learning_Style", palette="Blues_d", legend=False)

plt.title("Bar Plot - Average Final Exam Score per Learning Style", fontsize=15)
plt.xlabel("Learning Style")
plt.ylabel("Average Final Exam Score")

plt.tight_layout()
plt.show()

####################################################################################

#OBJECTIVE -4: Hypothesis Testing - Chi Square Test and T-Test

#T-Test -> Do Highly Engaged Students Perform Better?

df["Engagement_Category"] = pd.qcut(df["Engagement_Score"], q=3, labels=["Low", "Medium", "High"])

low_engagement = df[df["Engagement_Category"] == "Low"]["Final_Exam_Score"]
high_engagement = df[df["Engagement_Category"] == "High"]["Final_Exam_Score"]


t_stat, p_value = ttest_ind(low_engagement, high_engagement, equal_var=False)

print(f"T-Statistic: {t_stat:.2f}")
print(f"P-Value: {p_value:.5f}\n")

if p_value < 0.05:
    print("Reject H0: High engagement students perform significantly better.")
else:
    print("Fail to Reject H0: No significant difference in performance.")


#Chi-Square Test -> Does Engagement Level Affect Dropout?

# Convert Engagement Score into categories (Low, Medium, High)
df["Engagement_Category"] = pd.qcut(df["Engagement_Score"], q=3, labels=["Low", "Medium", "High"])

# Contingency table
contingency_table = pd.crosstab(df["Engagement_Category"], df["Dropout_Likelihood"])

# Perform Chi-Square Test
chi2_stat, p_value, dof, expected = stats.chi2_contingency(contingency_table)

print(f"Chi-Square Statistic: {chi2_stat:.2f}")
print(f"P-Value: {p_value:.5f}\n")

if p_value < 0.05:
    print("Reject H0: Engagement Level significantly affects Dropout Likelihood.")
else:
    print("Fail to Reject H0: No significant relationship between Engagement Level and Dropout.")

print(f"Chi-Square Statistic: {chi2_stat:.2f}")
print(f"P-Value: {p_value:.5f}")

#Grouped bar chart
contingency_table.plot(kind='bar', figsize=(9, 6), colormap='viridis')
plt.title("Dropout Likelihood by Engagement Level", fontsize=16)
plt.xlabel("Engagement Level", fontsize=12)
plt.ylabel("Number of Students", fontsize=12)
plt.xticks(rotation=0)
plt.legend(title="Dropout Likelihood")
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

#OBJECTIVE -5: Identify key factors influencing Dropout Likelihood using charts and plots

#Distribution Analysis (Box plot)
sns.set(style="whitegrid")

important_features = [
    'Time_Spent_on_Videos',
    'Assignment_Completion_Rate',
    'Forum_Participation',
    'Final_Exam_Score'
]

fig, axes = plt.subplots(1, 4, figsize=(16, 6))

sns.boxplot(y=df['Time_Spent_on_Videos'], ax=axes[0], color='skyblue')
axes[0].set_title("Time Spent on Videos")
axes[0].set_xticks([])

sns.boxplot(y=df['Assignment_Completion_Rate'], ax=axes[1], color='lightgreen')
axes[1].set_title("Assignment Completion Rate")
axes[1].set_xticks([])

sns.boxplot(y=df['Forum_Participation'], ax=axes[2], color='salmon')
axes[2].set_title("Forum Participation")
axes[2].set_xticks([])

sns.boxplot(y=df['Final_Exam_Score'], ax=axes[3], color='plum')
axes[3].set_title("Final Exam Score")
axes[3].set_xticks([])

plt.suptitle("Boxplots of Key Features", fontsize=16, fontweight='bold')
plt.tight_layout(rect=[0, 0.03, 1, 0.90])
plt.show()

# Visualization of categorical features
sns.set(style="whitegrid")
categorical_features = ['Gender', 'Education_Level', 'Learning_Style', 'Dropout_Likelihood']
plt.figure(figsize=(16, 10))
plt.subplot(2, 2, 1)
sns.countplot(x='Gender', data=df, hue="Gender", palette='pastel', legend=False)
plt.title('Gender Distribution')

plt.subplot(2, 2, 2)
sns.countplot(x='Education_Level', data=df, hue="Education_Level", palette='Set2', legend=False)
plt.title('Education Level Distribution')
plt.xticks(rotation=30)

plt.subplot(2, 2, 3)
sns.countplot(x='Learning_Style', data=df, hue="Learning_Style", palette='Set3', legend=False)
plt.title('Learning Style Distribution')
plt.xticks(rotation=30)

plt.subplot(2, 2, 4)
sns.countplot(x='Dropout_Likelihood', data=df, hue="Dropout_Likelihood", palette='coolwarm', legend=False)
plt.title('Dropout Likelihood Distribution')

plt.suptitle("Distribution of Categorical Features", fontsize=16, fontweight='bold')
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()

#Develop an Interactive Dashboard for Educators

# PIE CHART – Learning Style Distribution
pie_chart = px.pie(
    df, names="Learning_Style", title="Distribution of Learning Styles",
    color_discrete_sequence=px.colors.sequential.RdBu
)

# DONUT CHART – Dropout Distribution
donut_chart = px.pie(
    df, names="Dropout_Likelihood", title="Dropout Likelihood (Yes/No)",
    hole=0.5, color_discrete_sequence=px.colors.sequential.Mint
)

# Dash app layout
app = dash.Dash(__name__)

# Define a dark background style
dark_style = {
    'backgroundColor': '#1e1e2f',
    'color': '#f0f0f0',
    'fontFamily': 'Arial',
    'padding': '20px'
}

app.layout = html.Div(style={'backgroundColor': dark_style['backgroundColor']}, children=[
    html.H1("Personalized Learning Dashboard", style={
        'textAlign': 'center',
        'color': dark_style['color'],
        'paddingTop': '20px'
    }),

    html.Div([
        dcc.Graph(figure=pie_chart)
    ], style={'width': '48%', 'display': 'inline-block', 'padding': '10px'}),

    html.Div([
        dcc.Graph(figure=donut_chart)
    ], style={'width': '48%', 'display': 'inline-block', 'padding': '10px'}),

    html.Footer("Developed by Jyotsna Chaudhary", style={
        'textAlign': 'center',
        'padding': '20px',
        'color': dark_style['color'],
        'fontSize': '14px'
    })
])
    
if __name__ == '__main__':
    app.run_server(debug=True)
