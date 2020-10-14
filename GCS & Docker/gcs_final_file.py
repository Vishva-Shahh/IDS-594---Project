# -*- coding: utf-8 -*-
"""GCS_Final_File.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1aUhmePyPCvqEA8v2uTMaudXqHaGiTESC
"""

#Create bucket to store ML model
from google.cloud import storage
bucket_name = "trail_bucket"

storage_client = storage.Client()
storage_client.create_bucket(bucket_name)

for bucket in storage_client.list_buckets():
    print(bucket.name)

#Decision tree model
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
df = pd.read_csv("https://github.com/bgweber/Twitch/raw/master/Recommendations/games-expand.csv")
x = df.drop(['label'], axis=1)
y = df['label']
dtree = DecisionTreeClassifier()
dtree.fit(x,y)

#Store dtree
import pickle
pickle.dump(dtree, open("logit.pkl", 'wb'))
dtree = pickle.load(open("logit.pkl", 'rb'))
dtree.predict_proba(x)

#Saving the file to GCS
from google.cloud import storage
bucket_name = "trail_bucket"
storage_client = storage.Client()
bucket = storage_client.get_bucket(bucket_name)
blob = bucket.blob("serverless/logit/v1")
blob.upload_from_filename("logit.pkl")

#Running the code
import requests
result = requests.post("https://us-central1-clever-overview-289222.cloudfunctions.net/echo",json = { 'G1':'1', 'G2':'0', 'G3':'0', 'G4':'0', 'G5':'0', 'G6':'0', 'G7':'0', 'G8':'0', 'G9':'0', 'G10':'0'})
print(result.json())