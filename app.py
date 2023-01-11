import pickle
from datetime import datetime
from flask import Flask, request, render_template
from sklearn.feature_extraction.text import CountVectorizer

       
          

app = Flask(__name__)
with open("Winner_Model.pkl", "rb") as model:
    classification=pickle.load(model)



@app.route('/')
def home():
    return render_template('index.html')# renders index.html

@app.route('/predict',methods=['POST'])# gets the values that were sent to '/predict' by 'index.html'
def predict():
    
    text=[str(x) for x in request.form.values()]

    prediction = classification.predict(text)
    print(prediction[0])        

    return render_template('index.html', prediction_text=f'{str(prediction[0])}')# displays the prediction inside the '<b>{{ prediction_text }}</b>' that we've seen in 'index.html'

if __name__ == "__main__":
    app.run(debug=True)# Runs the Web App
            



