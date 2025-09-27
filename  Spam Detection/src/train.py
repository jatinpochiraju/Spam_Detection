import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from preprocessing import TextPreprocessor, FeatureExtractor
from models import SpamDetector, ModelComparison
import os

def create_sample_data():
    """Create sample spam/ham data for demonstration"""
    spam_messages = [
        "URGENT! You have won $1000000! Call now to claim your prize!",
        "FREE! Click here to get your free iPhone now! Limited time offer!",
        "Congratulations! You've been selected for a cash prize. Text CLAIM to 12345",
        "WINNER! You have won a luxury vacation. Call 1-800-SCAM now!",
        "Get rich quick! Make $5000 a day working from home!",
        "Your account will be suspended unless you verify immediately. Click here.",
        "Hot singles in your area want to meet you! Join now for free!",
        "Lose 30 pounds in 30 days with this miracle pill!",
        "You owe $500 in unpaid taxes. Pay immediately to avoid arrest.",
        "Claim your free gift card worth $500! Act now!"
    ]
    
    ham_messages = [
        "Hey, are we still meeting for lunch tomorrow?",
        "Thanks for the great presentation today. Well done!",
        "Can you pick up milk on your way home?",
        "The meeting has been moved to 3 PM in conference room B.",
        "Happy birthday! Hope you have a wonderful day.",
        "Don't forget about the dentist appointment at 2 PM.",
        "The weather looks great for our picnic this weekend.",
        "I'll be running 10 minutes late to our meeting.",
        "Could you send me the report when you get a chance?",
        "Great job on the project! The client loved it."
    ]
    
    # Create more samples by variations
    extended_spam = spam_messages * 5
    extended_ham = ham_messages * 5
    
    # Create DataFrame
    data = []
    for msg in extended_spam:
        data.append({'message': msg, 'label': 'spam'})
    for msg in extended_ham:
        data.append({'message': msg, 'label': 'ham'})
    
    return pd.DataFrame(data)

def load_data():
    """Load spam detection dataset"""
    # Try to load real dataset, fallback to sample data
    data_path = 'data/spam_dataset.csv'
    
    if os.path.exists(data_path):
        print("Loading dataset from file...")
        df = pd.read_csv(data_path)
    else:
        print("Creating sample dataset...")
        df = create_sample_data()
        
        # Save sample data
        os.makedirs('data', exist_ok=True)
        df.to_csv(data_path, index=False)
        print(f"Sample dataset saved to {data_path}")
    
    return df

def explore_data(df):
    """Explore and visualize the dataset"""
    print("\n" + "="*50)
    print("DATASET EXPLORATION")
    print("="*50)
    
    print(f"Dataset shape: {df.shape}")
    print(f"\nClass distribution:")
    print(df['label'].value_counts())
    
    # Create visualizations
    plt.figure(figsize=(12, 4))
    
    # Class distribution
    plt.subplot(1, 2, 1)
    df['label'].value_counts().plot(kind='bar', color=['skyblue', 'salmon'])
    plt.title('Class Distribution')
    plt.xlabel('Label')
    plt.ylabel('Count')
    plt.xticks(rotation=0)
    
    # Message length distribution
    plt.subplot(1, 2, 2)
    df['message_length'] = df['message'].str.len()
    df.boxplot(column='message_length', by='label', ax=plt.gca())
    plt.title('Message Length by Class')
    plt.suptitle('')
    
    plt.tight_layout()
    plt.savefig('data/data_exploration.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return df

def train_models(df):
    """Train and compare different models"""
    print("\n" + "="*50)
    print("TRAINING MODELS")
    print("="*50)
    
    # Initialize preprocessor and feature extractor
    preprocessor = TextPreprocessor()
    feature_extractor = FeatureExtractor(method='tfidf', max_features=3000)
    
    # Preprocess text
    print("Preprocessing text...")
    df['processed_message'] = df['message'].apply(preprocessor.preprocess)
    
    # Extract features
    print("Extracting features...")
    X = feature_extractor.fit_transform(df['processed_message'])
    y = df['label']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Training set size: {X_train.shape[0]}")
    print(f"Test set size: {X_test.shape[0]}")
    
    # Initialize model comparison
    comparison = ModelComparison()
    
    # Add different models
    models_to_test = ['naive_bayes', 'svm', 'random_forest', 'logistic_regression']
    
    for model_type in models_to_test:
        detector = SpamDetector(model_type=model_type)
        comparison.add_model(model_type, detector)
    
    # Train all models
    comparison.train_all(X_train, y_train)
    
    # Evaluate all models
    comparison.evaluate_all(X_test, y_test)
    
    # Print comparison results
    comparison.print_comparison()
    
    # Save best model
    best_model_name, best_score = comparison.get_best_model()
    best_model = comparison.models[best_model_name]
    
    os.makedirs('models', exist_ok=True)
    best_model.save_model(f'models/best_spam_detector.pkl')
    
    # Save feature extractor
    import joblib
    joblib.dump(feature_extractor, 'models/feature_extractor.pkl')
    joblib.dump(preprocessor, 'models/preprocessor.pkl')
    
    print(f"\nBest model ({best_model_name}) saved to models/best_spam_detector.pkl")
    
    return best_model, feature_extractor, preprocessor

def main():
    """Main training pipeline"""
    print("Starting Spam Detection Training Pipeline")
    print("="*60)
    
    # Load data
    df = load_data()
    
    # Explore data
    df = explore_data(df)
    
    # Train models
    best_model, feature_extractor, preprocessor = train_models(df)
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE!")
    print("="*60)
    print("Files created:")
    print("- models/best_spam_detector.pkl")
    print("- models/feature_extractor.pkl") 
    print("- models/preprocessor.pkl")
    print("- data/spam_dataset.csv")
    print("- data/data_exploration.png")

if __name__ == "__main__":
    main()