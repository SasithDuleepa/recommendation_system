from fastapi import FastAPI

import json
from unicodedata import name
from flask import Flask, jsonify, request
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
import pymongo
import requests

app = FastAPI()


@app.get("/")
async def root():

    # find from DB!!!
    myclient = pymongo.MongoClient("mongodb://localhost:27017/")
    mydb = myclient["v4u"]
    mycol = mydb["history"]
    # find input values for recommend
    y = list(mycol.find().sort("_id", -1).limit(5))
    print(y)

    index_list = []

    df_1 = pd.read_csv("book.csv")
    df = df_1[["authors", "title", "publisher", "isbn"]]
    df.isnull().sum()
    features = []
    for i in range(df.shape[0]):
        features.append(" ".join(list(df.iloc[i].values)))
    df["features"] = features
    cvec = CountVectorizer()
    cv_df = cvec.fit_transform(df["features"])
    cs = cosine_similarity(cv_df)

    outputlist = []
    outputlist.clear()

    for i in range(5):
        y1 = y[i]
        y2 = y1['title']

        def recommend(y2):
            movie_index = df[[y2 in name for name in df["title"]]].index[0]
            scores = list(enumerate(cs[movie_index]))
            sorted_scores = sorted(scores, key=lambda x: x[1], reverse=True)
            output = df_1.iloc[[h[0]
                                for h in sorted_scores[2:5]]][["title", "bookID"]]

            df2 = output["title"].tolist()
            print(output["title"].tolist())

            index_list.append(df2)
        recommend(y2)
    for sublist in index_list:
        outputlist.extend(sublist)

    #json_data = json.dumps(outlist)
    print(outputlist)
    return {"message": outputlist}
# uvicorn main:app --reload
