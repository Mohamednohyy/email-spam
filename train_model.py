import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Download required NLTK data
nltk.download('punkt')
nltk.download('stopwords')

class EmailSpamDetector:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=5000)
        self.model = MultinomialNB()
        self.stop_words = set(stopwords.words('english'))
        
    def preprocess_text(self, text):
        # Convert to lowercase
        text = text.lower()
        # Remove special characters and digits
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        # Tokenize
        tokens = word_tokenize(text)
        # Remove stopwords
        tokens = [t for t in tokens if t not in self.stop_words]
        return ' '.join(tokens)
    
    def load_and_preprocess_data(self, filepath):
        # Load the dataset
        data = pd.read_csv(filepath)
        
        # Enron dataset has 'Message' and 'Spam/Ham' columns
        # Rename columns to match our expected format
        data = data.rename(columns={'Message': 'text', 'Spam/Ham': 'label'})
        
        # Convert 'spam'/'ham' labels to binary format if needed
        data['label'] = data['label'].map({'spam': 1, 'ham': 0})
        
        # SIMPLE DATA CLEANING:
        
        # 1. Handle missing values
        data['text'] = data['text'].fillna('')
        
        # 2. Remove duplicates
        data = data.drop_duplicates(subset=['text'])
        
        # 3. Remove noise (very short messages)
        data = data[data['text'].str.len() >= 10]
        
        # 4. Standardize formatting (remove extra whitespace)
        data['text'] = data['text'].str.replace(r'\s+', ' ', regex=True).str.strip()
        
        # Preprocess the text data
        data['processed_text'] = data['text'].apply(self.preprocess_text)
        
        return data
    
    def train(self, X_train, y_train):
        # Transform text data to TF-IDF features
        X_train_tfidf = self.vectorizer.fit_transform(X_train)
        
        # Train the model
        self.model.fit(X_train_tfidf, y_train)
    
    def evaluate(self, X_test, y_test):
        # Transform test data
        X_test_tfidf = self.vectorizer.transform(X_test)
        
        # Make predictions
        y_pred = self.model.predict(X_test_tfidf)
        
        # Generate classification report
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        support = np.sum(y_test == 1)  # Number of spam samples in test set
        
        # Create metrics dictionary
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'support': support
        }
        
        # Generate text report for console output
        report = classification_report(y_test, y_pred)
        
        # Generate confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        
        # Save metrics to a text file for easy access
        with open('model_metrics.txt', 'w') as f:
            f.write(f"Accuracy: {accuracy:.4f}\n")
            f.write(f"Precision: {precision:.4f}\n")
            f.write(f"Recall: {recall:.4f}\n")
            f.write(f"F1-score: {f1:.4f}\n")
            f.write(f"Support: {support}\n")
        
        return report, cm, metrics
    
    def save_model(self, model_path='spam_detector_model.pkl', metrics=None):
        # Save both the vectorizer and model along with metrics
        joblib.dump({
            'vectorizer': self.vectorizer,
            'model': self.model,
            'metrics': metrics
        }, model_path)
    
    def plot_confusion_matrix(self, cm):
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.savefig('confusion_matrix.png')
        plt.close()

def main():
    # Initialize the detector
    detector = EmailSpamDetector()
    
    # Load and preprocess data
    # Note: Replace 'spam_dataset.csv' with your actual dataset file
    data = detector.load_and_preprocess_data('enron_spam_data.csv')
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        data['processed_text'],
        data['label'],
        test_size=0.2,
        random_state=42
    )
    
    # Train the model
    detector.train(X_train, y_train)
    
    # Evaluate the model
    report, cm, metrics = detector.evaluate(X_test, y_test)
    
    # Print evaluation results
    print("Classification Report:")
    print(report)
    
    # Print specific metrics
    print("\nModel Metrics:")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1-score: {metrics['f1_score']:.4f}")
    print(f"Support: {metrics['support']}")
    
    # Plot and save confusion matrix
    detector.plot_confusion_matrix(cm)
    
    # Save the model with metrics
    detector.save_model(metrics=metrics)
    
    print("Model training completed and saved successfully!")

if __name__ == "__main__":
    main() 