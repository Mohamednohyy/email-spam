import streamlit as st
import joblib
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re

# Try to use NLTK data without downloading
try:
    # Just check if the data is available
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
except LookupError:
    st.warning("NLTK data not found. Please run download_nltk_data.py first.")

# Set page config
st.set_page_config(
    page_title="Email Spam Detector",
    page_icon="📧",
    layout="centered"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stTextArea textarea {
        height: 200px;
    }
    .prediction-box {
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    </style>
""", unsafe_allow_html=True)

def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    # Remove special characters and digits
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Tokenize
    tokens = word_tokenize(text)
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [t for t in tokens if t not in stop_words]
    return ' '.join(tokens)

def load_model():
    try:
        model_data = joblib.load('spam_detector_model.pkl')
        return model_data['vectorizer'], model_data['model'], model_data.get('metrics', None)
    except:
        st.error("Model file not found. Please train the model first.")
        return None, None, None

def main():
    st.title("📧 Email Spam Detector")
    
    # Load the model and metrics
    vectorizer, model, metrics = load_model()
    
    # Display model metrics
    st.subheader("Model Performance Metrics")
    
    # Display metrics if available
    if metrics is not None:
        # Create columns for metrics
        col1, col2, col3 = st.columns(3)
        col4, col5 = st.columns(2)
        
        with col1:
            st.metric("Accuracy", f"{metrics['accuracy']:.4f}")
        with col2:
            st.metric("Precision", f"{metrics['precision']:.4f}")
        with col3:
            st.metric("Recall", f"{metrics['recall']:.4f}")
        with col4:
            st.metric("F1-Score", f"{metrics['f1_score']:.4f}")
        with col5:
            st.metric("Support", f"{metrics['support']}")
    else:
        st.info("Model metrics not available. Please train the model first.")
    
    # Display confusion matrix
    try:
        st.image("confusion_matrix.png", caption="Confusion Matrix")
    except:
        st.info("Confusion matrix not available.")
    
    st.write("Enter an email text below to check if it's spam or not.")

    # Use model for prediction
    if vectorizer is not None and model is not None:
        # Text input
        email_text = st.text_area("Email Content", height=200)
        
        if st.button("Check for Spam"):
            if email_text:
                # Preprocess the text
                processed_text = preprocess_text(email_text)
                
                # Transform the text
                text_features = vectorizer.transform([processed_text])
                
                # Make prediction
                prediction = model.predict(text_features)[0]
                probabilities = model.predict_proba(text_features)[0]
                spam_prob = probabilities[1]
                ham_prob = probabilities[0]
                
                # Display result
                if prediction == 1:
                    st.markdown(
                        f'<div class="prediction-box" style="background-color: #ffcdd2;">'
                        f'<h3>🚫 Spam Detected!</h3>'
                        f'<p>Confidence: {spam_prob:.2%}</p>'
                        f'</div>',
                        unsafe_allow_html=True
                    )
                else:
                    st.markdown(
                        f'<div class="prediction-box" style="background-color: #c8e6c9;">'
                        f'<h3>✅ Not Spam</h3>'
                        f'<p>Confidence: {ham_prob:.2%}</p>'
                        f'</div>',
                        unsafe_allow_html=True
                    )
                
                # Display classification results
                st.subheader("Classification Results")
                
                # Create two columns for the probabilities
                col1, col2 = st.columns(2)
                
                # Display probabilities in columns
                with col1:
                    st.metric("Spam Probability", f"{spam_prob:.2%}")
                
                with col2:
                    st.metric("Not Spam Probability", f"{ham_prob:.2%}")
                
                # Display the tokens that influenced the classification
                st.subheader("Key Words Detected")
                tokens = processed_text.split()
                if tokens:
                    st.write(", ".join(tokens[:10]))
                else:
                    st.write("No significant words detected after preprocessing.")
                
                # Display model's decision explanation
                st.subheader("Decision Explanation")
                if prediction == 1:
                    st.write("This email was classified as spam because it contains patterns commonly found in spam messages.")
                else:
                    st.write("This email was classified as legitimate because it doesn't contain typical spam patterns.")
                
            else:
                st.warning("Please enter some text to analyze.")

if __name__ == "__main__":
    main() 