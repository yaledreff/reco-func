import logging

import azure.functions as func
import pandas as pd
import json

import numpy as np

from scipy.sparse import csr_matrix
import sklearn
from sklearn.preprocessing import *
from scipy.sparse.linalg import svds
from io import BytesIO


def setMinArticlesViews(dfArticlesPerActiveUser, minViews=2):
    # Retire les articles qui n'ont pas un minimum de vues de 'minViews'
    dfArticlesViews = dfArticlesPerActiveUser[['click_article_id', 'session_id']].groupby(['click_article_id']).count().reset_index()
    lstArticlesOut = dfArticlesViews[dfArticlesViews['session_id'] <= minViews]['click_article_id'].unique().tolist()
    dfArticlesPerActiveUser = dfArticlesPerActiveUser[~dfArticlesPerActiveUser['click_article_id'].isin(lstArticlesOut)]
    return dfArticlesPerActiveUser

def setMinUsersViews(dfArticlesPerActiveUser, minViews=2):
    # Retire les utilisatuers qui n'ont pas un minimum de consultations de 'minViews'
    dfUsersViews = dfArticlesPerActiveUser[['user_id', 'session_id']].groupby(['user_id']).count().reset_index()
    lstUsersOut = dfUsersViews[dfUsersViews['session_id'] < minViews]['user_id'].unique().tolist()
    dfArticlesPerActiveUser = dfArticlesPerActiveUser[~dfArticlesPerActiveUser['user_id'].isin(lstUsersOut)]
    return dfArticlesPerActiveUser

def getSVDFactoMatrix(dfArticlesPerActiveUser):
    dfUsersItemsPivotMatrix = dfArticlesPerActiveUser.pivot(index='user_id', columns='click_article_id', values='session_id').fillna(0)
    usersItemsPivotMatrix = dfUsersItemsPivotMatrix.values
    usersIds = list(dfUsersItemsPivotMatrix.index)
    #Factorisation de la matrice
    usersItemsPivotSparseMatrix = csr_matrix(usersItemsPivotMatrix)
    U, sigma, Vt = svds(usersItemsPivotSparseMatrix, k=15)
    sigma = np.diag(sigma)
    allUserPredictedRatings = np.dot(np.dot(U, sigma), Vt)
    allUserPredictedRatingsNorm = normalize(allUserPredictedRatings)
    DfPreds = pd.DataFrame(allUserPredictedRatingsNorm, columns=dfUsersItemsPivotMatrix.columns, index=usersIds).transpose()
    return DfPreds


def main(req: func.HttpRequest, inFileUser, outFilePred: func.Out[str]) -> func.HttpResponse:

    fileBytes = inFileUser.read() 
    fileBytesIO = BytesIO(fileBytes)    
    dfArticlesPerActiveUser = pd.read_csv(fileBytesIO)
    dfArticlesPerActiveUser = setMinArticlesViews(dfArticlesPerActiveUser, minViews=100)
    dfArticlesPerActiveUser = setMinUsersViews(dfArticlesPerActiveUser, minViews=60)
    dfPreds = getSVDFactoMatrix(dfArticlesPerActiveUser)

    dfCSV = dfPreds.to_csv()
    outFilePred.set(dfCSV)

    return func.HttpResponse(json.dumps("Entrainement terminé avec succès"))

