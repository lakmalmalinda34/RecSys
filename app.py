from flask import Flask, request, jsonify
from tensorflow.keras import models, layers, utils  #(2.6.0)
import numpy as np
import pandas as pd
from datetime import datetime
import json

def prediction(userID, weekend, daytime):
    # Load the trained model
    model = models.load_model('Model/model_V1.h5')
    dtf_products = pd.read_excel("Data/contents.xlsx")
    dtf_products = dtf_products[~dtf_products["Category"].isna()]
    dtf_products["product"] = range(0,len(dtf_products))
    dtf_products = dtf_products[["product","author", "Category"]].set_index("product")
    
    columns = dtf_products['Category'].unique().tolist()
    for col in columns:
        dtf_products[col] = dtf_products["Category"].apply(lambda x: 1 if col in x and x != "" else 0)

    columns = dtf_products['author'].unique().tolist()
    for col in columns:
        dtf_products[col] = dtf_products["author"].apply(lambda x: 1 if col in x and x != "" else 0)

    features = dtf_products.drop(["author", "Category"], axis=1).columns
    context = ['daytime', 'weekend']

    Prediction_df = dtf_products.drop(["author", "Category"], axis=1).reset_index()
    input_data = {
        "user": [userID for _ in range(0,len(dtf_products))],
        "weekend":  [weekend for _ in range(0,len(dtf_products))],
        "daytime": [daytime for _ in range(0,len(dtf_products))]
            }
    input_df = pd.DataFrame(input_data)
    Fin_pre = Prediction_df.merge(input_df, how="left", left_on="product", right_index=True)
    Fin_pre["yhat"] = model.predict([Fin_pre["user"], Fin_pre["product"], Fin_pre[features], Fin_pre[context]])
    sorted_Fin_pre_df = Fin_pre.sort_values(by='yhat', ascending=False)
    product_lst=list(sorted_Fin_pre_df["product"].head(5))
    ContentID_lst = [item + 1 for item in product_lst]

    return ContentID_lst



app = Flask(__name__)

# API route for predictions
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json  # Get JSON data from POST request
                
        user_index = int(data['user_index'])
        context_weekend = int(data['weekend'])
        context_daytime = int(data['daytime'])

        recommended_contents = prediction(user_index, context_weekend, context_daytime)
        return jsonify({"recommended_ContentIds": recommended_contents})

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(debug=True)
