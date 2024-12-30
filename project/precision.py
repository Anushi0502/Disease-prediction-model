import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.model_selection import train_test_split
import joblib,os
import pandas as pd

def load_symptoms_data():
        # Load symptoms from cache if available
        if os.path.exists('symptoms_cache.pkl'):
            return joblib.load('symptoms_cache.pkl')
        else:
            df_symptoms = pd.read_csv('Symptom-severity.csv')
            joblib.dump(df_symptoms, 'symptoms_cache.pkl')
            return df_symptoms
def clean_and_encode_data(df):
        df.fillna(0, inplace=True)
        df = df.applymap(lambda x: x.lower().strip() if isinstance(x, str) else x)
        df_symptoms=load_symptoms_data()
        symptoms = df_symptoms['Symptom'].unique()
        for symptom in symptoms:
            weight = df_symptoms.loc[df_symptoms['Symptom'] == symptom, 'weight'].values[0]
            df.replace(symptom, weight, inplace=True)

        df.replace(['dischromic _patches', 'spotting_ urination', 'foul_smell_of urine'], 0, inplace=True)
        return df
# Load the model
model = joblib.load("ensemble_model_parallel.joblib")

# Load the dataset (assuming the dataset is in CSV format)
df = pd.read_csv("dataset.csv")
df = clean_and_encode_data(df)
print(df)
# Prepare data (adjust based on your actual dataset)
x = df.iloc[:, 1:].values
y = df['Disease'].values

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Make predictions
y_pred = model.predict(X_test)

# Calculate precision, recall, and F1 score
precision = precision_score(y_test, y_pred, average='macro')  # Adjust average for multi-class if needed
recall = recall_score(y_test, y_pred, average='macro')
f1 = f1_score(y_test, y_pred, average='macro')

# Print scores below the graph
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1 Score: {f1:.2f}")

# Create bar chart for the scores
metrics = ['Precision', 'Recall', 'F1 Score']
scores = [precision, recall, f1]

plt.figure(figsize=(8, 6))
plt.bar(metrics, scores, color=['blue', 'green', 'orange'])
plt.ylim(0, 1)
plt.title('Model Performance Metrics')
plt.ylabel('Score')
plt.xlabel('Metrics')

for i, score in enumerate(scores):
            plt.text(i, score + 0.02, f"{score:.2f}", ha='center', fontsize=12)

# Show the plot
plt.show()
