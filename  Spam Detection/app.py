from flask import Flask, render_template, request, jsonify
import os
import sys

# Add src directory to path
sys.path.append('src')

from predict import SpamPredictor

app = Flask(__name__)

# Initialize predictor
try:
    predictor = SpamPredictor()
    predictor_loaded = True
except Exception as e:
    print(f"Warning: Could not load predictor: {e}")
    predictor_loaded = False

@app.route('/')
def index():
    """Main page"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """API endpoint for predictions"""
    if not predictor_loaded:
        return jsonify({
            'error': 'Model not loaded. Please train the model first.',
            'success': False
        })
    
    try:
        data = request.get_json()
        message = data.get('message', '')
        
        if not message.strip():
            return jsonify({
                'error': 'Please provide a message to analyze.',
                'success': False
            })
        
        # Make prediction
        result = predictor.predict_single(message)
        
        return jsonify({
            'success': True,
            'prediction': result['prediction'],
            'confidence': round(result['confidence'] * 100, 2),
            'spam_probability': round(result['spam_probability'] * 100, 2),
            'ham_probability': round(result['ham_probability'] * 100, 2)
        })
        
    except Exception as e:
        return jsonify({
            'error': f'Prediction failed: {str(e)}',
            'success': False
        })

@app.route('/batch_predict', methods=['POST'])
def batch_predict():
    """API endpoint for batch predictions"""
    if not predictor_loaded:
        return jsonify({
            'error': 'Model not loaded. Please train the model first.',
            'success': False
        })
    
    try:
        data = request.get_json()
        messages = data.get('messages', [])
        
        if not messages:
            return jsonify({
                'error': 'Please provide messages to analyze.',
                'success': False
            })
        
        # Make predictions
        results = predictor.predict_batch(messages)
        
        # Format results
        formatted_results = []
        for result in results:
            formatted_results.append({
                'message': result['message'],
                'prediction': result['prediction'],
                'confidence': round(result['confidence'] * 100, 2),
                'spam_probability': round(result['spam_probability'] * 100, 2)
            })
        
        return jsonify({
            'success': True,
            'results': formatted_results
        })
        
    except Exception as e:
        return jsonify({
            'error': f'Batch prediction failed: {str(e)}',
            'success': False
        })

if __name__ == '__main__':
    # Create templates directory if it doesn't exist
    os.makedirs('templates', exist_ok=True)
    
    # Check if template exists, create if not
    template_path = 'templates/index.html'
    if not os.path.exists(template_path):
        print("Creating web interface template...")
        create_template()
    
    print("Starting Flask application...")
    print("Open http://localhost:5001 in your browser")
    app.run(debug=True, host='0.0.0.0', port=5001)

def create_template():
    """Create HTML template for web interface"""
    html_content = '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Spam Detection</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            max-width: 800px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }
        .container {
            background-color: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        h1 {
            color: #333;
            text-align: center;
            margin-bottom: 30px;
        }
        textarea {
            width: 100%;
            height: 120px;
            padding: 15px;
            border: 2px solid #ddd;
            border-radius: 5px;
            font-size: 16px;
            resize: vertical;
        }
        button {
            background-color: #007bff;
            color: white;
            padding: 12px 30px;
            border: none;
            border-radius: 5px;
            font-size: 16px;
            cursor: pointer;
            margin-top: 15px;
        }
        button:hover {
            background-color: #0056b3;
        }
        .result {
            margin-top: 20px;
            padding: 20px;
            border-radius: 5px;
            font-size: 16px;
        }
        .spam {
            background-color: #f8d7da;
            border: 1px solid #f5c6cb;
            color: #721c24;
        }
        .ham {
            background-color: #d4edda;
            border: 1px solid #c3e6cb;
            color: #155724;
        }
        .error {
            background-color: #fff3cd;
            border: 1px solid #ffeaa7;
            color: #856404;
        }
        .loading {
            display: none;
            text-align: center;
            margin-top: 20px;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🛡️ Spam Detection System</h1>
        <p>Enter a message below to check if it's spam or legitimate (ham):</p>
        
        <textarea id="messageInput" placeholder="Enter your message here..."></textarea>
        <br>
        <button onclick="predictSpam()">Analyze Message</button>
        
        <div class="loading" id="loading">
            <p>Analyzing message...</p>
        </div>
        
        <div id="result"></div>
        
        <div style="margin-top: 40px;">
            <h3>Sample Messages to Try:</h3>
            <ul>
                <li><a href="#" onclick="setMessage('Congratulations! You have won $1000! Click here to claim now!')">Spam Example</a></li>
                <li><a href="#" onclick="setMessage('Hey, are we still meeting for lunch tomorrow?')">Ham Example</a></li>
                <li><a href="#" onclick="setMessage('URGENT: Your account will be suspended. Verify immediately!')">Suspicious Example</a></li>
            </ul>
        </div>
    </div>

    <script>
        function setMessage(message) {
            document.getElementById('messageInput').value = message;
        }
        
        function predictSpam() {
            const message = document.getElementById('messageInput').value.trim();
            const resultDiv = document.getElementById('result');
            const loadingDiv = document.getElementById('loading');
            
            if (!message) {
                resultDiv.innerHTML = '<div class="result error">Please enter a message to analyze.</div>';
                return;
            }
            
            // Show loading
            loadingDiv.style.display = 'block';
            resultDiv.innerHTML = '';
            
            // Make API call
            fetch('/predict', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({message: message})
            })
            .then(response => response.json())
            .then(data => {
                loadingDiv.style.display = 'none';
                
                if (data.success) {
                    const prediction = data.prediction;
                    const confidence = data.confidence;
                    const spamProb = data.spam_probability;
                    
                    const resultClass = prediction === 'spam' ? 'spam' : 'ham';
                    const emoji = prediction === 'spam' ? '🚨' : '✅';
                    
                    resultDiv.innerHTML = `
                        <div class="result ${resultClass}">
                            <h3>${emoji} Prediction: ${prediction.toUpperCase()}</h3>
                            <p><strong>Confidence:</strong> ${confidence}%</p>
                            <p><strong>Spam Probability:</strong> ${spamProb}%</p>
                        </div>
                    `;
                } else {
                    resultDiv.innerHTML = `<div class="result error">Error: ${data.error}</div>`;
                }
            })
            .catch(error => {
                loadingDiv.style.display = 'none';
                resultDiv.innerHTML = `<div class="result error">Error: ${error.message}</div>`;
            });
        }
        
        // Allow Enter key to submit
        document.getElementById('messageInput').addEventListener('keypress', function(e) {
            if (e.key === 'Enter' && e.ctrlKey) {
                predictSpam();
            }
        });
    </script>
</body>
</html>'''
    
    with open('templates/index.html', 'w') as f:
        f.write(html_content)