import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
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

# Step 4: Train the classifier
classifier = SVC()
classifier.fit(X_train_features, y_train)

# Save the trained model
joblib.dump(classifier, 'FolderFinal/FolderFinal/SVM_spam_ham_classifier.joblib')

# Streamlit UI for message classification
st.title("Spam and Ham Message Classification")
message = st.text_input("Enter a message:")
classify_button = st.button("Classify")

if classify_button:
    if message:
        # Step 5: Evaluate the classifier
        y_pred = classifier.predict(X_test_features)

        accuracy = accuracy_score(y_test, y_pred)
        st.write("Accuracy:", accuracy)

        message_features = vectorizer.transform([message])
        prediction = classifier.predict(message_features)
        predicted_label = "spam" if prediction[0] == 1 else "ham"
        st.write(f"<p style='font-size: 20px; font-weight: bold;'>Classified message is: {predicted_label}</p>", unsafe_allow_html=True)

        # Plot the label distribution as a bar graph
        filtered_data = data[data['label_num'] == prediction[0]]
        label_counts = filtered_data['label_num'].value_counts()
        labels = ['ham', 'spam']
        values = [label_counts.get(0, 0), label_counts.get(1, 0)]

        fig, ax = plt.subplots()
        ax.bar(labels, values)
        ax.set_xlabel('Label')
        ax.set_ylabel('Count')
        ax.set_title('Label Distribution')
        st.pyplot(fig)

        # # Confusion matrix
        # cm = confusion_matrix(y_test, y_pred)
        # st.write("Confusion Matrix:")
        # cm_df = pd.DataFrame(cm, columns=labels, index=labels)

        # # Modify table styling with thick border
        # cm_table = f"<table style='border: thick solid black;'>{cm_df.to_html()}</table>"
        # st.write(cm_table, unsafe_allow_html=True)

        # # Heatmap of confusion matrix
        # fig, ax = plt.subplots()
        # sns.heatmap(cm_df, annot=True, cmap='Blues', fmt='d')
        # ax.set_xlabel('Predicted Labels')
        # ax.set_ylabel('True Labels')
        # ax.set_title('Confusion Matrix')
        # st.pyplot(fig)

        # Line graph
        x_values = np.arange(len(labels))
        y_values = [accuracy, 1 - accuracy]

        fig, ax = plt.subplots()
        ax.plot(x_values, y_values, marker='o')
        ax.set_xticks(x_values)
        ax.set_xticklabels(labels)
        ax.set_xlabel('Label')
        ax.set_ylabel('Accuracy')
        ax.set_title('Accuracy by Label')

        st.pyplot(fig)
