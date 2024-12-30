import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QFormLayout, QPushButton, QVBoxLayout, QComboBox,
    QTextEdit, QMessageBox, QHBoxLayout, QStatusBar, QProgressBar
)
import joblib
import os
import qtawesome as qta

# Color Constants
VIVID_RED = "#FF3D00"
DARK_PURPLE = "#4682B4"
LUMINOUS_YELLOW = "#FFFF00"
RICH_BLACK = "#2F4F4F"
NEON_BLUE = "#00BFFF"

class PredictionThread(QThread):
    prediction_made = pyqtSignal(str)

    def __init__(self, model, symptom_weights):
        super().__init__()
        self.model = model
        self.symptom_weights = symptom_weights

    def run(self):
        try:
            prediction = self.model.predict([self.symptom_weights])
            print(prediction)
            self.prediction_made.emit(prediction[0])
        except Exception as e:
            self.prediction_made.emit(f"Error during prediction: {str(e)}")

class DiseasePredictionApp(QWidget):
    def __init__(self):
        super().__init__()
        self.symptoms_list = []
        self.user_input = [None] * 6
        self.df_symptoms = self.load_symptoms_data()
        self.symptoms = self.df_symptoms['Symptom'].unique().tolist()
        self.model = self.load_model()
        self.init_ui()

    def load_symptoms_data(self):
        if os.path.exists('symptoms_cache.pkl'):
            return joblib.load('symptoms_cache.pkl')
        else:
            df_symptoms = pd.read_csv('Symptom-severity.csv')
            joblib.dump(df_symptoms, 'symptoms_cache.pkl')
            return df_symptoms

    def get_symptom_weight(self, symptom):
        symptom_dict = dict(zip(self.df_symptoms['Symptom'].str.lower().str.strip(), self.df_symptoms['weight']))
        return symptom_dict.get(symptom.lower().strip(), 0)

    def init_ui(self):
        self.setWindowFlags(Qt.FramelessWindowHint)
        self.setGeometry(200, 80, 1000, 600)
        self.setWindowTitle('Disease Prediction App')
        self.setStyleSheet(f"background-color: {RICH_BLACK};")

        layout = QVBoxLayout()
        title_label = QLabel("Disease Prediction From Symptoms")
        title_label.setFont(QFont('Arial', 36, QFont.Bold))
        title_label.setStyleSheet(f"color: {LUMINOUS_YELLOW};")
        layout.addWidget(title_label, alignment=Qt.AlignCenter)

        form_layout = QFormLayout()

        for i in range(5):
            self.setup_combo_box(form_layout, f"Symptom {i + 1}", self.symptoms)

        button_layout = QHBoxLayout()
    
        predict_button = QPushButton(qta.icon('fa.check'), " Predict")
        predict_button.setFixedSize(200, 50)
        predict_button.setStyleSheet(f"""
            background-color: {VIVID_RED}; 
            color: white; 
            font-size: 32px; 
            border-radius: 8px;
        """)
        predict_button.clicked.connect(self.predict_disease)

        close_button = QPushButton(qta.icon('fa.times'), " Close")
        close_button.setFixedSize(200, 50)
        close_button.setStyleSheet(f"""
            background-color: {VIVID_RED}; 
            color: white; 
            font-size: 32px; 
            border-radius: 8px;
        """)
        close_button.clicked.connect(sys.exit)

        button_layout.addWidget(predict_button)
        button_layout.addSpacing(20)  # Add space between buttons
        button_layout.addWidget(close_button)

        layout.addLayout(form_layout)
        layout.addSpacing(20)  # Add space above buttons
        layout.addLayout(button_layout)

        result_layout = QHBoxLayout()
        result_label = QLabel("Prediction Result:")
        result_label.setFont(QFont('Arial', 32, QFont.Bold))
        result_label.setStyleSheet(f"color: {LUMINOUS_YELLOW};")
        result_layout.addWidget(result_label)

        self.result_text = QTextEdit()
        self.result_text.setReadOnly(True)
        self.result_text.setStyleSheet(f"background-color: {DARK_PURPLE}; color: {LUMINOUS_YELLOW}; font-size: 32px;")
        result_layout.addWidget(self.result_text)

        layout.addLayout(result_layout)

        self.status_bar = QStatusBar()
        self.status_bar.setStyleSheet(f"background-color: {DARK_PURPLE}; color: {LUMINOUS_YELLOW}; font-size: 32px;")
        self.progress_bar = QProgressBar(self)
        self.progress_bar.setRange(0, 0)  # Indeterminate mode
        self.progress_bar.setStyleSheet(f"background-color: {VIVID_RED};")
        layout.addWidget(self.progress_bar)
        layout.addWidget(self.status_bar)

        self.setLayout(layout)

    def setup_combo_box(self, layout, label_text, options):
        label = QLabel(label_text)
        label.setFont(QFont('Arial', 32))
        label.setStyleSheet(f"color: {LUMINOUS_YELLOW};")

        combo_box = QComboBox()
        combo_box.setFont(QFont('Arial', 32))
        combo_box.setStyleSheet(f"background-color: {DARK_PURPLE}; color: {LUMINOUS_YELLOW};")
        combo_box.addItems(options)
        
        layout.addRow(label, combo_box)
        self.symptoms_list.append(combo_box)

    def predict_disease(self):
        self.result_text.clear()
        self.progress_bar.setVisible(True)
        self.status_bar.showMessage("Processing...")

        for i in range(5):
            current_text = self.symptoms_list[i].currentText()
            if current_text:
                self.user_input[i] = current_text
            else:
                self.show_error_message("Please enter all symptoms properly.")
                return

        self.perform_prediction()

    def show_error_message(self, message):
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Warning)
        msg.setText(message)
        msg.setWindowTitle("Input Error")
        msg.exec_()

    def perform_prediction(self):
        symptoms = self.user_input[:5]
        symptom_weights = np.array([self.get_symptom_weight(s) for s in symptoms])
        encoded_symptoms = np.concatenate((symptom_weights, np.zeros(17 - len(symptom_weights))))  # Zero padding

        self.thread = PredictionThread(self.model, encoded_symptoms)
        self.thread.prediction_made.connect(self.display_prediction)
        self.thread.start()
        self.thread.finished.connect(lambda: self.progress_bar.setVisible(False))

    def display_prediction(self, prediction):
        self.result_text.setText(f"The most probable disease is: {prediction}")
        self.status_bar.showMessage("Prediction complete.")

    def load_model(self):
        if os.path.exists("disease_prediction_model.joblib"):
            return joblib.load("disease_prediction_model.joblib")
        else:
            return self.train_and_save_model()

    def train_and_save_model(self):
        df = pd.read_csv('dataset.csv')
        df = self.clean_and_encode_data(df)
        print(df)
        data = df.iloc[:, 1:].values
        labels = df['Disease'].values

        x_train, x_test, y_train, y_test = train_test_split(data, labels, train_size=0.85, shuffle=True)
        model = SVC(kernel='rbf', C=1.0, gamma='scale')  # Optimized model with RBF kernel
        model.fit(x_train, y_train)

        preds = model.predict(x_test)
        conf_mat = confusion_matrix(y_test, preds)
        plt.figure(figsize=(10, 6))
        sns.heatmap(conf_mat, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.show()

        precision = precision_score(y_test, preds, average='macro')
        recall = recall_score(y_test, preds, average='macro')
        f1 = f1_score(y_test, preds, average='macro')

        print(f"Precision: {precision:.2f}")
        print(f"Recall: {recall:.2f}")
        print(f"F1 Score: {f1:.2f}")

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

        plt.savefig("model_metrics.png")
        plt.show()

        joblib.dump(model, "disease_prediction_model.joblib")
        return model

    def clean_and_encode_data(self,df):
        df.fillna(0, inplace=True)
        df = df.applymap(lambda x: x.lower().strip() if isinstance(x, str) else x)
        symptoms = self.df_symptoms['Symptom'].unique()
        for symptom in symptoms:
            weight = self.df_symptoms.loc[self.df_symptoms['Symptom'] == symptom, 'weight'].values[0]
            df.replace(symptom, weight, inplace=True)

        df.replace(['dischromic _patches', 'spotting_ urination', 'foul_smell_of urine'], 0, inplace=True)
        return df

app = QApplication(sys.argv)
window = DiseasePredictionApp()
if __name__ == "__main__":
    window.show()
    sys.exit(app.exec_())
