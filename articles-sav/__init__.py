import logging

import azure.functions as func
import pandas as pd
import json
import os
import shutil

def main(req: func.HttpRequest, outFileArticles: func.Out[str]) -> func.HttpResponse:
    logging.info('Coucou')
    pFile = req.files["file"]
    clear_text = pFile.read().decode('utf-8')
    outFileArticles.set(clear_text)

    return func.HttpResponse(json.dumps("articles mis à jour"))