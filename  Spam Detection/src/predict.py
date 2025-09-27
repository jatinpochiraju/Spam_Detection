import joblib
import os
from preprocessing import TextPreprocessor, FeatureExtractor
from models import SpamDetector

class SpamPredictor:
    def __init__(self, model_path='models/best_spam_detector.pkl'):
        self.model_path = model_path
        self.model = None
        self.preprocessor = None
        self.feature_extractor = None
        self.load_components()
    
    def load_components(self):
        """Load all required components"""
        try:
            # Load model
            self.model = joblib.load(self.model_path)
            
            # Load preprocessor and feature extractor
            self.preprocessor = joblib.load('models/preprocessor.pkl')
            self.feature_extractor = joblib.load('models/feature_extractor.pkl')
            
            print("All components loaded successfully!")
            
        except FileNotFoundError as e:
            print(f"Error loading components: {e}")
            print("Please run training first: python src/train.py")
            raise
    
    def predict_single(self, message):
        """Predict if a single message is spam or ham"""
        # Preprocess the message
        processed_message = self.preprocessor.preprocess(message)
        
        # Extract features
        features = self.feature_extractor.transform([processed_message])
        
        # Make prediction
        prediction = self.model.predict(features)[0]
        probability = self.model.predict_proba(features)[0]
        
        # Get confidence score
        if prediction == 'spam':
            confidence = probability[1]  # Probability of spam
        else:
            confidence = probability[0]  # Probability of ham
        
        return {
            'message': message,
            'prediction': prediction,
            'confidence': confidence,
            'spam_probability': probability[1],
            'ham_probability': probability[0]
        }
    
    def predict_batch(self, messages):
        """Predict multiple messages"""
        results = []
        for message in messages:
            result = self.predict_single(message)
            results.append(result)
        return results
    
    def print_prediction(self, result):
        """Print prediction result in a formatted way"""
        print(f"\nMessage: {result['message']}")
        print(f"Prediction: {result['prediction'].upper()}")
        print(f"Confidence: {result['confidence']:.2%}")
        print(f"Spam Probability: {result['spam_probability']:.2%}")
        print(f"Ham Probability: {result['ham_probability']:.2%}")
        print("-" * 50)

def test_predictions():
    """Test the predictor with sample messages"""
    predictor = SpamPredictor()
    
    test_messages = [
        "Congratulations! You've won $1000! Click here to claim now!",
        "Hey, can you pick me up at 5 PM?",
        "URGENT: Your account will be closed. Verify now!",
        "Thanks for the meeting today. See you tomorrow.",
        "Get rich quick! Make money fast with this secret method!",
        "Don't forget about dinner with mom on Sunday.",
        "FREE iPhone! Limited time offer! Act now!",
        "The project deadline has been extended to next Friday."
    ]
    
    print("SPAM DETECTION PREDICTIONS")
    print("=" * 60)
    
    for message in test_messages:
        result = predictor.predict_single(message)
        predictor.print_prediction(result)

if __name__ == "__main__":
    test_predictions()