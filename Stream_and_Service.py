import random
import threading
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone

import requests
import uvicorn
from fastapi import FastAPI, HTTPException
from river import anomaly, stats

# model and features
features = ["temperature", "humidity", "noise_level"]

model = anomaly.HalfSpaceTrees(
    n_trees=25,
    height=12,
    window_size=250,
    seed=42
)

tresh = stats.RollingQuantile(
    q = 0.95,
    window_size=250
)


# generate normal sensor data
def normal_sample():
    return {
        "temperature": random.normalvariate(70, 2),
        "humidity": random.normalvariate(60, 2),
        "noise_level": random.normalvariate(30, 2),
    }


# generate abnormal sensor data
def anomaly_sample():
    return {
        "temperature": random.normalvariate(80, 3),
        "humidity": random.normalvariate(70, 3) + random.uniform(1,45), # add some randon number to make this over 100 at random to make invalid sensor data
        "noise_level": random.normalvariate(36, 3),
    }


# data validation
def validate_data(data: dict):

    # check all required values are present
    for feature in features:

        if feature not in data:
            return False

        # check values are not null
        if data[feature] is None:
            return False

        # check values are numeric
        if not isinstance(data[feature], (int, float)):
            return False

        # normalize value by dividing by 100
        normalized_value = data[feature] / 100

        # check normalized value is between 0 and 1
        if normalized_value < 0 or normalized_value >= 1:
            return False

    return True


# normalize sensor data
def normalize_data(data: dict):

    return {
        feature: data[feature] / 100
        for feature in features
    }


# train model with normal data before stream starts and build up 0.95 threshold
def train_model():

    # generate normal training data
    training_data = [
        normal_sample()
        for _ in range(250)
    ]

    print("Starting Trainig of Model and Threshold")

    # train model one data point at a time
    for data in training_data:

        # normalize sensor data
        x = normalize_data(data)

        # score
        score_observation = model.score_one(x)

        # update border
        tresh.update(score_observation)

        # learn
        model.learn_one(x)
        

    print("Model and Threshold Training finished")


# prediction - return anomaly score
def predict_one(data: dict):

    # normalize sensor data
    x = normalize_data(data)

    # calculate anomaly score 
    score = model.score_one(x)

    # get actual 0.95 threshold
    threshold = tresh.get()
    
    if threshold is not None and score < threshold:

        # learn to model
        model.learn_one(x)
        # update threshold
        tresh.update()
    else: None  
        
    # return prediction score   
    return float(score)


# main part - whole streaming and predicting 
def stream_loop():

    # wait until FastAPI is fully started
    time.sleep(2)

    while True:

        # generate sensor data
        # 10% of the observations are intentionally abnormal 
        # for testing reasons, could be ANOMAL data or even invalid sensor data
        if random.random() < 0.10:
            data = anomaly_sample()
            data_type = "ANOMALY"
        else:
            data = normal_sample()
            data_type = "NORMAL" 

        # add timestamp for data point
        data["timestamp"] = datetime.now(timezone.utc).isoformat()

        try:

            # validate sensor data
            validation = requests.post(
                "http://127.0.0.1:8000/validate",
                json=data,
                timeout=1,
            )

            validation_result = validation.json()

            if not validation_result["valid"]:
                print("Invalid sensor data")
                time.sleep(0.5)
                continue # break loop start next iteration

            # send validated data to prediction
            r = requests.post(
                "http://127.0.0.1:8000/predict",
                json=data,
                timeout=1,
            )

            r.raise_for_status()

            result = r.json()

            # print source, sensor data and anomaly score
            print(
                f"TIME={data['timestamp']} "
                f"SOURCE : {data_type:7} "
                f"T={data['temperature']:.2f} "
                f"H={data['humidity']:.2f} "
                f"N={data['noise_level']:.2f} "
                f"-> score={result['score']:.10f}"
            )

        except requests.RequestException as e:

            print("Request failed:", e)

        time.sleep(0.5)


# lifespan - manages FastAPI from start to end
@asynccontextmanager
async def lifespan(app: FastAPI):

    # train model before starting stream
    train_model()

    # create background worker to run stream_loop, ends when Main is closed
    thread = threading.Thread(
        target=stream_loop,
        daemon=True,
    )

    thread.start()

    print("Stream started")

    yield

    print("Shutting down")


# FastAPI, build up API
app = FastAPI(lifespan=lifespan)


# data validation in front of model
@app.post("/validate")
def validate(data: dict):

    valid = validate_data(data)

    return {
        "valid": valid
    }


# model as a service
@app.post("/predict")
def predict(data: dict):

    # validate data before prediction for double security
    if not validate_data(data):
        raise HTTPException(
            status_code=400,
            detail="Invalid Data has passed valid check - Check valid check"
        )

    # prediction only happens after final validation
    score = predict_one(data)

    return {
        "score": score
    }


# run all
if __name__ == "__main__":

    uvicorn.run(
        app,
        host="127.0.0.1",
        port=8000,
        log_level="warning"
    )

