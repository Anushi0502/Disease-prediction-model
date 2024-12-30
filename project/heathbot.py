# <h1><b>Disease Prediction with GUI<b></h1>
# A disease prediction model working on support vector machine (SVM). It takes the symptoms of the user as input along with its location and predicts the most probable disease which the user might be facing. The same data can be sent to cloud and being later analysed using analytical tool tableau.
# The data has been taken from https://www.kaggle.com/itachi9604/disease-symptom-description-dataset.
# <h2>Importing the libraries</h2>
import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
import seaborn as sns
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QFormLayout, QPushButton, QVBoxLayout, QComboBox,
    QTextEdit, QMessageBox, QHBoxLayout, QStatusBar, QProgressBar
)
import joblib
import os
import qtawesome as qta
from qt_material import apply_stylesheet  # Importing Qt Material stylesheet

class PredictionThread(QThread):
    prediction_made = pyqtSignal(str)

    def __init__(self, model, symptom_weights):
        super().__init__()
        self.model = model
        self.symptom_weights = symptom_weights

    def run(self):
        prediction = self.model.predict([self.symptom_weights])
        self.prediction_made.emit(prediction[0])

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
        symptom_dict = dict(zip(self.df_symptoms['Symptom'], self.df_symptoms['weight']))
        return symptom_dict.get(symptom, 0)

    def init_ui(self):
        self.setWindowFlags(Qt.FramelessWindowHint)
        self.setGeometry(200, 100, 1000, 600)
        self.setWindowTitle('Disease Prediction App')

        layout = QVBoxLayout()
        title_label = QLabel("Disease Prediction From Symptoms")
        title_label.setStyleSheet("font-size: 36px; font-weight: bold; color: #333;")
        layout.addWidget(title_label, alignment=Qt.AlignCenter)

        form_layout = QFormLayout()
        locations_options = ["New Delhi", "Mumbai", "Chennai", "Kolkata", "Bengaluru"]

        for i in range(5):
            self.setup_combo_box(form_layout, f"Symptom {i + 1}", self.symptoms)
    
        self.setup_combo_box(form_layout, "Location", locations_options)

        button_layout = QHBoxLayout()
    
        predict_button = QPushButton(qta.icon('fa.check'), " Predict")
        predict_button.setFixedSize(200, 50)
        predict_button.clicked.connect(self.predict_disease)
    
        close_button = QPushButton(qta.icon('fa.times'), " Close")
        close_button.setFixedSize(200, 50)
        close_button.clicked.connect(self.close)

        button_layout.addWidget(predict_button)
        button_layout.addSpacing(20)
        button_layout.addWidget(close_button)

        layout.addLayout(form_layout)
        layout.addSpacing(20)
        layout.addLayout(button_layout)

        result_layout = QHBoxLayout()
        result_label = QLabel("Prediction Result:")
        result_label.setStyleSheet("font-size: 24px; font-weight: bold;")
        result_layout.addWidget(result_label)

        self.result_text = QTextEdit()
        self.result_text.setReadOnly(True)
        result_layout.addWidget(self.result_text)

        layout.addLayout(result_layout)

        self.status_bar = QStatusBar()
        self.progress_bar = QProgressBar(self)
        self.progress_bar.setRange(0, 0)  # Indeterminate mode
        layout.addWidget(self.progress_bar)
        layout.addWidget(self.status_bar)

        self.setLayout(layout)

    def setup_combo_box(self, layout, label_text, options):
        label = QLabel(label_text)
        label.setStyleSheet("font-size: 16px; color: #555;")
        combo_box = QComboBox()
        combo_box.setStyleSheet("font-size: 16px;")
        combo_box.addItems(options)
        layout.addRow(label, combo_box)
        self.symptoms_list.append(combo_box)

    def predict_disease(self):
        self.result_text.clear()
        self.progress_bar.setVisible(True)
        self.status_bar.showMessage("Processing...")

        for i in range(6):
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
        encoded_symptoms = np.concatenate((symptom_weights, np.zeros(17 - len(symptom_weights))))

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
        df = pd.read_csv('dataset/dataset.csv')
        df_symptoms = pd.read_csv("dataset/symptom_Description.csv", index_col="Disease")
        df = self.clean_and_encode_data(df)

        data = df.iloc[:, 1:].values
        labels = df['Disease'].values

        x_train, x_test, y_train, y_test = train_test_split(data, labels, train_size=0.85, shuffle=True)
        model = SVC()
        model.fit(x_train, y_train)

        preds = model.predict(x_test)
        conf_mat = confusion_matrix(y_test, preds)
        sns.heatmap(conf_mat)

        joblib.dump(model, "disease_prediction_model.joblib")
        return model

    def clean_and_encode_data(self, df):
        df.fillna(0, inplace=True)
        symptoms = self.df_symptoms['Symptom'].unique()
        for symptom in symptoms:
            weight = self.df_symptoms.loc[self.df_symptoms['Symptom'] == symptom, 'weight'].values[0]
            df.replace(symptom, weight, inplace=True)

        df.replace(['dischromic _patches', 'spotting_ urination', 'foul_smell_of urine'], 0, inplace=True)
        return df

if __name__ == "__main__":
    app = QApplication(sys.argv)
    apply_stylesheet(app, theme='dark_teal.xml')  # Applying Qt Material theme
    window = DiseasePredictionApp()
    window.show()
    sys.exit(app.exec_())
