from flask import Flask, render_template, request
from src.exception import CustomException
from src.logger import logging
from src.pipeline.prediction_pipeline import PredictPipeline, CustomData

application = Flask(__name__)
app = application


@app.route("/",methods = ["GET","POST"])
def predict_datapoint():
    if request.method == "GET":
        return render_template("index.html")

    else:
        data = CustomData(
            LIMIT_BAL = request.form.get("LIMIT_BAL"),
            SEX = request.form.get("SEX"),
            EDUCATION = request.form.get("EDUCATION"),
            MARRIAGE = request.form.get("MARRIAGE"),
            AGE = request.form.get("AGE"),
            PAY_0 = request.form.get("PAY_0"),
            PAY_2 = request.form.get("PAY_2"),
            PAY_3 = request.form.get("PAY_3"),
            PAY_4 = request.form.get("PAY_4"),
            PAY_5 = request.form.get("PAY_5"),
            PAY_6 = request.form.get("PAY_6"),
            BILL_AMT1 = request.form.get("BILL_AMT1"),
            BILL_AMT2 = request.form.get("BILL_AMT2"),
            BILL_AMT3 = request.form.get("BILL_AMT3"),
            BILL_AMT4 = request.form.get("BILL_AMT4"),
            BILL_AMT5 = request.form.get("BILL_AMT5"),
            BILL_AMT6 = request.form.get("BILL_AMT6"),
            PAY_AMT1 = request.form.get("PAY_AMT1"),
            PAY_AMT2 = request.form.get("PAY_AMT2"),
            PAY_AMT3 = request.form.get("PAY_AMT3"),
            PAY_AMT4 = request.form.get("PAY_AMT4"),
            PAY_AMT5 = request.form.get("PAY_AMT5"),
            PAY_AMT6 = request.form.get("PAY_AMT6")
            )
        
        final_data = data.get_data_as_data_frame()
        predict_pipeline = PredictPipeline()

        prediction = predict_pipeline.predict(final_data)

        result = prediction.tolist()


        if result[0] == 1:
            string = "The credit card holder will be Defaulter in the next month"
            final_result_class = 'negative'
        else:
            string = "The Credit card holder will NOT be Defaulter in the next month" 
            final_result_class = 'positive'

        print(string)

        return render_template('result.html', final_result=string, final_result_class=final_result_class)




if __name__=="__main__":
    app.run(host="0.0.0.0",debug=True)