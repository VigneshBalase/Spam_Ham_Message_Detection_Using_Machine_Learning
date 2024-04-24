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

        # Plot the label distribution as a bar graph for Naive Bayes
        nb_filtered_data = data[data['label_num'] == nb_prediction[0]]
        nb_label_counts = nb_filtered_data['label_num'].value_counts()
        nb_labels = ['ham', 'spam']
        nb_values = [nb_label_counts.get(0, 0), nb_label_counts.get(1, 0)]

        # Plot the label distribution as a bar graph for Random Forest
        rf_filtered_data = data[data['label_num'] == rf_prediction[0]]
        rf_label_counts = rf_filtered_data['label_num'].value_counts()
        rf_labels = ['ham', 'spam']
        rf_values = [rf_label_counts.get(0, 0), rf_label_counts.get(1, 0)]

        # Plot the label distribution as a bar graph for SVM
        svm_filtered_data = data[data['label_num'] == svm_prediction[0]]
        svm_label_counts = svm_filtered_data['label_num'].value_counts()
        svm_labels = ['ham', 'spam']
        svm_values = [svm_label_counts.get(0, 0), svm_label_counts.get(1, 0)]

        # Line graph for Naive Bayes accuracy
        nb_label = 'Naive Bayes'

        # Line graph for Random Forest accuracy
        rf_label = 'Random Forest'

        # Line graph for SVM accuracy
        svm_label = 'SVM'

        # Create a single figure with subplots
        fig, axs = plt.subplots(2, 3, figsize=(12, 8))

        # Bar graph for Naive Bayes label distribution
        axs[0, 0].bar(nb_labels, nb_values)
        axs[0, 0].set_xlabel('Label')
        axs[0, 0].set_ylabel('Count')
        axs[0, 0].set_title('Naive Bayes: Label Distribution')

        # Bar graph for Random Forest label distribution
        axs[0, 1].bar(rf_labels, rf_values)
        axs[0, 1].set_xlabel('Label')
        axs[0, 1].set_ylabel('Count')
        axs[0, 1].set_title('Random Forest: Label Distribution')

        # Bar graph for SVM label distribution
        axs[0, 2].bar(svm_labels, svm_values)
        axs[0, 2].set_xlabel('Label')
        axs[0, 2].set_ylabel('Count')
        axs[0, 2].set_title('SVM: Label Distribution')

        # Line graph for Naive Bayes accuracy
        axs[1, 0].plot([nb_label], [nb_accuracy], marker='o')
        axs[1, 0].set_xlabel('Classifier')
        axs[1, 0].set_ylabel('Accuracy')
        axs[1, 0].set_title('Naive Bayes: Accuracy')

        # Line graph for Random Forest accuracy
        axs[1, 1].plot([rf_label], [rf_accuracy], marker='o')
        axs[1, 1].set_xlabel('Classifier')
        axs[1, 1].set_ylabel('Accuracy')
        axs[1, 1].set_title('Random Forest: Accuracy')

        # Line graph for SVM accuracy
        axs[1, 2].plot([svm_label], [svm_accuracy], marker='o')
        axs[1, 2].set_xlabel('Classifier')
        axs[1, 2].set_ylabel('Accuracy')
        axs[1, 2].set_title('SVM: Accuracy')

        plt.tight_layout()
        st.pyplot(fig)
