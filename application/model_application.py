# Import the dependencies we need to run the code.
import os
import requests
import json
import gradio as gr
import numpy as np

# Get a few environment variables. These are so we:
# - Know what endpoint we should request
# - Set server name and port for Gradio
URL = os.getenv("INFERENCE_ENDPOINT")                       # You need to manually set this with an environment variable
GRADIO_SERVER_PORT = int(os.getenv("GRADIO_SERVER_PORT"))   # Automatically set by the Dockerfile
GRADIO_SERVER_NAME = os.getenv("GRADIO_SERVER_NAME")        # Automatically set by the Dockerfile

# Create a small function that sends data to the inference endpoint and recieves a prediction
def predict(distance_from_home,distance_from_last_transaction,ratio_to_median_purchase_price,repeat_retailer,used_chip,used_pin_number,online_order):
    payload = {
        "inputs": [
            {
                "name": "dense_input", 
                "shape": [1, 7], 
                "datatype": "FP32",
                "data": [[distance_from_home,distance_from_last_transaction,ratio_to_median_purchase_price,repeat_retailer,used_chip,used_pin_number,online_order]]
            },
            ]
        }
    headers = {
        'content-type': 'application/json'
    }

    response = requests.post(URL, json=payload, headers=headers)
    prediction = response.json()['outputs'][0]['data'][0]

    return "Fraud" if prediction >=0.995 else "Not fraud"


# Create and launch a Gradio interface that uses the prediction function to predict an output based on the inputs. 
# We also set up a few example inputs to make it easier to try out the application.
demo = gr.Interface(
    fn=predict, 
    inputs=["number","number","number","number","number","number","number"], 
    outputs="textbox",
    examples=[
        [57.87785658389723,0.3111400080477545,1.9459399775518593,1.0,1.0,0.0,0.0],
        [15.694985541059943,175.98918151972342,0.8556228290724207,1.0,0.0,0.0,1.0],
        [10.829942699255545,0.17559150228166587,1.2942188106198573,1.0,0.0,0.0,0.0],
        [5.091079490616996,0.8051525945853258,0.42771456119427587,1.0,0.0,0.0,1.0],
        [2.2475643282963613,5.600043547,0.36266257805709584,1.0,1.0,0.0,1.0],
        [44.19093600261837,0.5664862680583477,2.2227672978404707,1.0,1.0,0.0,1.0],
        [5.586407674186407,13.261073268058121,0.064768465,1.0,0.0,0.0,0.0],
        [3.7240191247148107,0.9568379284821842,0.27846494490815554,1.0,0.0,0.0,1.0],
        [4.8482465722805665,0.3207354272228163,1.2730495235601782,1.0,0.0,1.0,0.0],
        [0.8766322564943629,2.5036089266921437,1.5169993152858177,0.0,0.0,0.0,0.0]
        ],
    title="Predict Credit Card Fraud"
    )

demo.launch(server_name=GRADIO_SERVER_NAME, server_port=GRADIO_SERVER_PORT)
