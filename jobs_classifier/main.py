import fastapi
import pandas as pd

import model_setup
import enums
from text_preprocessing.preprocessing import Preprocess

from typing import Union
from fastapi import FastAPI

app = FastAPI()


@app.get("/")
def read_root():
    return {"USA Jobs Location Negotiable Classifier": "Given a set of USA Jobs descriptions, use this classifier to predict whether the locations for the jobs are negotiable or not."}


@app.get("/predict/{job_descriptions}")
def predict(job_descriptions: str):
    descriptions = [job_descriptions]

    clean_text = [Preprocess(t).text.lower() for t in descriptions]

    tfidf_module = model_setup.tfidf_trainer(pretrained_model=True)
    tfidf_out = tfidf_module.transform(clean_text)

    model = model_setup.JobSummaryClassifier(load_pretrained=True)
    predictions = model.predict(tfidf_out)[:, 1]

    predictions_with_confidence = []

    # always print out predictions
    for ix, (t, pred) in enumerate(zip(descriptions, predictions)):

        if pred >= enums.THRESHOLD:
            if pred >= enums.HIGH_CONFIDENCE:
                predictions_with_confidence.append(
                    f"\n{t} -------------> \n [GREEN]: Location Negotiable EGLIBLE with high confidence")
            else:
                predictions_with_confidence.append(
                    f"\n{t} -------------> \n [YELLOW]: Location Negotiable EGLIBLE medium confidence")

        else:
            if 1-pred >= enums.HIGH_CONFIDENCE:
                predictions_with_confidence.append(
                    f"\n{t} -------------> \n [RED]: Location Negotiable INEGLIBLE with high confidence")
            else:
                predictions_with_confidence.append(
                    f"\n{t} -------------> \n [YELLOW]: Location Negotiable INEGLIBLE with medium confidence")

    return {"predictions": predictions_with_confidence}
