from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import joblib
import os

class SpamDetector:
    def __init__(self, model_type='naive_bayes'):
        self.model_type = model_type
        self.model = self._get_model(model_type)
        self.is_trained = False
    
    def _get_model(self, model_type):
        """Initialize model based on type"""
        models = {
            'naive_bayes': MultinomialNB(alpha=1.0),
            'svm': SVC(kernel='linear', probability=True, random_state=42),
            'random_forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'logistic_regression': LogisticRegression(random_state=42, max_iter=1000)
        }
        
        if model_type not in models:
            raise ValueError(f"Model type {model_type} not supported")
        
        return models[model_type]
    
    def train(self, X_train, y_train):
        """Train the model"""
        self.model.fit(X_train, y_train)
        self.is_trained = True
        return self
    
    def predict(self, X):
        """Make predictions"""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        return self.model.predict(X)
    
    def predict_proba(self, X):
        """Get prediction probabilities"""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        return self.model.predict_proba(X)
    
    def evaluate(self, X_test, y_test):
        """Evaluate model performance"""
        y_pred = self.predict(X_test)
        
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, pos_label='spam'),
            'recall': recall_score(y_test, y_pred, pos_label='spam'),
            'f1_score': f1_score(y_test, y_pred, pos_label='spam')
        }
        
        return metrics, classification_report(y_test, y_pred)
    
    def save_model(self, filepath):
        """Save trained model"""
        if not self.is_trained:
            raise ValueError("Model must be trained before saving")
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump(self.model, filepath)
    
    def load_model(self, filepath):
        """Load trained model"""
        self.model = joblib.load(filepath)
        self.is_trained = True
        return self

class ModelComparison:
    def __init__(self):
        self.models = {}
        self.results = {}
    
    def add_model(self, name, model):
        """Add model to comparison"""
        self.models[name] = model
    
    def train_all(self, X_train, y_train):
        """Train all models"""
        for name, model in self.models.items():
            print(f"Training {name}...")
            model.train(X_train, y_train)
    
    def evaluate_all(self, X_test, y_test):
        """Evaluate all models"""
        for name, model in self.models.items():
            print(f"Evaluating {name}...")
            metrics, report = model.evaluate(X_test, y_test)
            self.results[name] = {
                'metrics': metrics,
                'report': report
            }
    
    def get_best_model(self, metric='f1_score'):
        """Get best performing model based on metric"""
        if not self.results:
            raise ValueError("No evaluation results available")
        
        best_score = 0
        best_model = None
        
        for name, result in self.results.items():
            score = result['metrics'][metric]
            if score > best_score:
                best_score = score
                best_model = name
        
        return best_model, best_score
    
    def print_comparison(self):
        """Print comparison results"""
        print("\n" + "="*60)
        print("MODEL COMPARISON RESULTS")
        print("="*60)
        
        for name, result in self.results.items():
            metrics = result['metrics']
            print(f"\n{name.upper()}:")
            print(f"  Accuracy:  {metrics['accuracy']:.4f}")
            print(f"  Precision: {metrics['precision']:.4f}")
            print(f"  Recall:    {metrics['recall']:.4f}")
            print(f"  F1-Score:  {metrics['f1_score']:.4f}")
        
        best_model, best_score = self.get_best_model()
        print(f"\nBest Model: {best_model} (F1-Score: {best_score:.4f})")