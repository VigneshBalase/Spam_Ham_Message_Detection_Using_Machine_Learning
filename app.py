import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix
import joblib

# Step 1: Load the dataset from CSV
data = pd.read_csv('FolderFinal/FolderFinal/FinalDataset/spam_ham_dataset1.csv')

# Step 2: Split the dataset into training and testing sets
X = data['text']  # Assuming 'text' is the column containing the features
y = data['label_num']  # Assuming 'label_num' is the column containing the target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)  # Set random state for reproducibility

# Step 3: Extract features from text data
vectorizer = CountVectorizer()
X_train_features = vectorizer.fit_transform(X_train)
X_test_features = vectorizer.transform(X_test)

# Step 4: Train the Naive Bayes classifier
nb_classifier = MultinomialNB()
nb_classifier.fit(X_train_features, y_train)

# Save the trained Naive Bayes model
joblib.dump(nb_classifier, 'FolderFinal/FolderFinal/NaiveBayes_spam_ham_classifier.joblib')

# Step 4: Train the Random Forest classifier
rf_classifier = RandomForestClassifier()
rf_classifier.fit(X_train_features, y_train)

# Save the trained Random Forest model
joblib.dump(rf_classifier, 'FolderFinal/FolderFinal/RFspam_ham_classifier.joblib')

# Step 4: Train the SVM classifier
svm_classifier = SVC()
svm_classifier.fit(X_train_features, y_train)

# Save the trained SVM model
joblib.dump(svm_classifier, 'FolderFinal/FolderFinal/SVM_spam_ham_classifier.joblib')

# Streamlit UI for message classification
st.title("Spam and Ham Message Classification")
message = st.text_input("Enter a message:")
classify_button = st.button("Classify")

if classify_button:
    if message:
        # Step 5: Evaluate the Naive Bayes classifier
        nb_y_pred = nb_classifier.predict(X_test_features)
        nb_accuracy = accuracy_score(y_test, nb_y_pred)

        # Step 5: Evaluate the Random Forest classifier
        rf_y_pred = rf_classifier.predict(X_test_features)
        rf_accuracy = accuracy_score(y_test, rf_y_pred)

        # Step 5: Evaluate the SVM classifier
        svm_y_pred = svm_classifier.predict(X_test_features)
        svm_accuracy = accuracy_score(y_test, svm_y_pred)

        st.write("Naive Bayes Accuracy:", nb_accuracy)
        st.write("Random Forest Accuracy:", rf_accuracy)
        st.write("SVM Accuracy:", svm_accuracy)

        message_features = vectorizer.transform([message])

        # Predict using Naive Bayes classifier
        nb_prediction = nb_classifier.predict(message_features)
        nb_predicted_label = "spam" if nb_prediction[0] == 1 else "ham"
        st.write(f"<p style='font-size: 20px; font-weight: bold;'>Naive Bayes: Classified message is {nb_predicted_label}</p>", unsafe_allow_html=True)

        # Predict using Random Forest classifier
        rf_prediction = rf_classifier.predict(message_features)
        rf_predicted_label = "spam" if rf_prediction[0] == 1 else "ham"
        st.write(f"<p style='font-size: 20px; font-weight: bold;'>Random Forest: Classified message is {rf_predicted_label}</p>", unsafe_allow_html=True)

        # Predict using SVM classifier
        svm_prediction = svm_classifier.predict(message_features)
        svm_predicted_label = "spam" if svm_prediction[0] == 1 else "ham"
        st.write(f"<p style='font-size: 20px; font-weight: bold;'>SVM: Classified message is {svm_predicted_label}</p>", unsafe_allow_html=True)

        # Plot the label distribution as a bar graph
        filtered_data = data[data['label_num'] == nb_prediction[0]]
        label_counts = filtered_data['label_num'].value_counts()
        labels = ['ham', 'spam']
        values = [label_counts.get(0, 0), label_counts.get(1, 0)]

        fig, ax = plt.subplots()
        ax.bar(labels, values)
        ax.set_xlabel('Label')
        ax.set_ylabel('Count')
        ax.set_title('Label Distribution')
        st.pyplot(fig)

        # Line graph
        labels = ['Naive Bayes', 'Random Forest', 'SVM']
        accuracies = [nb_accuracy, rf_accuracy, svm_accuracy]

        fig, ax = plt.subplots()
        ax.plot(labels, accuracies, marker='o')
        ax.set_xlabel('Classifier')
        ax.set_ylabel('Accuracy')
        ax.set_title('Accuracy by Classifier')

        st.pyplot(fig)
