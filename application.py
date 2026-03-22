from flask import Flask, request, render_template
import pandas as pd
import pickle

application = Flask(__name__)
app = application

ridge_model = pickle.load(open("models/ridgr.pkl", "rb"))
standard_scaler = pickle.load(open("models/scaler.pkl", "rb"))

@app.route("/")
def index():
    return render_template("home.html")

@app.route("/predict", methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        try:
            input_dict = {
                'Temperature': [float(request.form['Temperature'])],
                'RH': [float(request.form['RH'])],
                'Ws': [float(request.form['Ws'])],
                'Rain': [float(request.form['Rain'])],
                'FFMC': [float(request.form['FFMC'])],
                'DMC': [float(request.form['DMC'])],
                'DC': [float(request.form['DC'])],
                'ISI': [float(request.form['ISI'])],
                'BUI': [float(request.form['BUI'])],
                'Classes': [float(request.form['Classes'])],
                'region': [float(request.form['Region'])] 
            }

            input_df = pd.DataFrame(input_dict)
            new_data_scaled = standard_scaler.transform(input_df)
            prediction = ridge_model.predict(new_data_scaled)

            return render_template('index.html', result=round(prediction[0], 2))

        except Exception as e:
            return f"Error: {str(e)}"
    
    return render_template("index.html", result=None)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)