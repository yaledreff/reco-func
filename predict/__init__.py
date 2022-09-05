import logging

import azure.functions as func
import pandas as pd
import json

import numpy as np

from scipy.sparse import csr_matrix
import sklearn
from sklearn.preprocessing import *
from scipy.sparse.linalg import svds
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse

import pickle

from io import BytesIO

class CFRecommender:
    MODEL_NAME = 'Collaborative Filtering'

    def __init__(self, dfPredictions):
        self.dfPredictions = dfPredictions

    def get_model_name(self):
        return self.MODEL_NAME

    def recommend_items(self, userId, itemsToIgnore=[], topn=10):
        # Get and sort the user's predictions
        sortedUserPredictions = self.dfPredictions[userId].sort_values(ascending=False) \
            .reset_index().rename(columns={userId: 'recStrength'})
        # Recommend the highest predicted rating movies that the user hasn't seen yet.
        dfRecommendations = sortedUserPredictions[~sortedUserPredictions['click_article_id'].isin(itemsToIgnore)] \
            .sort_values('recStrength', ascending=False) \
            .head(topn)
        return dfRecommendations

def get_items_interacted(user_id, dfUserActivity):
    # On récupere les id d'articles consultés pour un utilisateur donnée.
    interacted_items = dfUserActivity.loc[user_id]['click_article_id']
    return set(interacted_items if type(interacted_items) == pd.Series else [interacted_items])

class ResponsePred():
    articleId: int
    score: float
    def __init__(self, articleId, score):
        self.articleId = articleId
        self.score = score
    def setPred(self, articleId, score):
        self.articleId = articleId
        self.score = score
    def getarticleId(self):
        return self.articleId
    def getScore(self):
        return self.score

class ResponsePreds():
    preds = []
    def __init__(self, dfPreds):
        self.preds = self.create(dfPreds)
    def getPreds(self):
        return self.preds
    def addPred(self, pred):
        self.preds.append(pred)
    def create(self, dfPreds):
        lstPreds = dfPreds.values.tolist()
        response = [ResponsePred(int(articleId), score) for articleId,  score in lstPreds]
        return response


def main(req: func.HttpRequest, inFileUser, inFilePred) -> func.HttpResponse:
    # lecture des paramètres contenus dans le corps Json de la requête
    pBody =  req.get_body()
    jBody = json.loads(pBody) 
    userId = jBody["userId"]
    topN = jBody["topN"]
    # chargement des données : matrice de factorisation SVD [Modèle collaboratif]
    fileBytes = inFileUser.read() 
    fileBytesIO = BytesIO(fileBytes)    
    dfArticlesPerActiveUser = pd.read_csv(fileBytesIO)
    fileBytes = inFilePred.read() 
    fileBytesIO = BytesIO(fileBytes)    
    dfPreds = pd.read_csv(fileBytesIO)
    dfPreds = dfPreds.set_index('click_article_id')
    dfPreds.columns = dfPreds.columns.astype(int)
    # Instanciation de la classe de recommandation (collaborative model)
    cfRecommenderModel = CFRecommender(dfPreds)
    # Liste les articles déjà lus par l'utilisateurs (exclus des recommandations)
    itemsToIgnore = get_items_interacted(userId, dfArticlesPerActiveUser)
    # prediction
    pred = cfRecommenderModel.recommend_items(userId, itemsToIgnore, topN)
    res = ResponsePreds(pred).getPreds()
    json_encoder = jsonable_encoder(res)
    resJson = JSONResponse(content=json_encoder)
    return func.HttpResponse(resJson.body.decode("utf-8"))


