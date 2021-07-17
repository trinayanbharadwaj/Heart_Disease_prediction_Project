import numpy as np
import pickle
from flask import Flask, request, render_template

app = Flask(__name__)
model = pickle.load(open("model.pkl", "rb"))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict",methods=["POST"])
def predict():
    features = [int(x) for x in request.form.values()]
    features_np = [np.array(features)]
    probability = model.predict_proba(features_np)[:,1]
    prob = np.round(probability*100,2)
    prediction_prob = np.where(probability>=0.40, 1, 0)
    
    if prediction_prob == 0:
        result = "Good News! You are less likely to have heart disease in the next 10 years. You have {} % chance of heart disease.".format(prob)
    elif prediction_prob == 1:
       result = "Our model has predicted that you might have heart issue in the upcoming 10 years.  You have {} % chance of heart disease.".format(prob) 
        
    return render_template("index.html", predict=result)  

if __name__ == "__main__":
    app.run(debug=True)                  
