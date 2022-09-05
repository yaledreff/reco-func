import logging
from multiprocessing import context

import azure.functions as func
import pandas as pd
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse

import json
from io import BytesIO

def getListUsers(dfArticlesPerActiveUser):
    lstUsers = dfArticlesPerActiveUser['user_id'].unique().tolist()
    return lstUsers

def main(req: func.HttpRequest, inFileUser) -> func.HttpResponse:

    fileBytes = inFileUser.read() 
    fileBytesIO = BytesIO(fileBytes)    
    dfArticlesPerActiveUser = pd.read_csv(fileBytesIO)
    lstUsers = getListUsers(dfArticlesPerActiveUser)
    responseJson = jsonable_encoder(lstUsers)
    return func.HttpResponse(json.dumps(lstUsers))

