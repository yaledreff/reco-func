import logging

import azure.functions as func
import pandas as pd
import json
import os
import shutil

def main(req: func.HttpRequest, outFileUser: func.Out[str]) -> func.HttpResponse:
    logging.info('Coucou')
    pFile = req.files["file"]
    clear_text = pFile.read().decode('utf-8')
    outFileUser.set(clear_text)

    return func.HttpResponse(json.dumps("utilisateurs mis à jour"))

