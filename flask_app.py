from flask import Flask, render_template, request, jsonify  
import joblib  

app = Flask(__name__)  

# Cargar el modelo entrenado  
model = joblib.load('spam_model.pkl')  

@app.route('/')  
def index():  
    return render_template('index.html')  

@app.route('/predict', methods=['POST'])  
def predict():  
    email_text = request.json['text']  
    prediction = model.predict([email_text])[0]  
    result = 'Spam' if prediction == 1 else 'No Spam'  
    return jsonify({'prediction': result})  

if __name__ == "__main__":  
    app.run(debug=True) 