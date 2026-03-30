"""
Ce script permet d'inférer le model de machine learning et de le mettre à disposition
dans un Webservice. Il pourra donc être utilisé par notre chatbot par exemple,
ou directement par un front. Remplir ce script une fois l'entrainement du model fonctionne
"""

import os
import pickle

# DONE: Importer les dépendances utiles au bon développement en Python (dataclass, enum, pandas)
# DONE : Importer les dépendances pour sérialiser / désérialiser le model
from dataclasses import dataclass
from enum import Enum
import pandas as pd
# DONE : Importer les dépendances fastAPI
from fastapi import FastAPI,Depends

# TODO : Importer les dépendances OTEL pour le monitoring

from titanic.api.auth import verify_token


JAEGER_ENDPOINT = os.getenv("JAEGER_ENDPOINT", "http://jaeger.willemanmariepro-dev.svc.cluster.local:4318/v1/traces")

# TODO : Intégrer les configurations d'OTEL et instancier le tracer. Peut être fait plus tard si le cours
# sur l'observabilité n'est pas encore donné

# DONE : Instancier l'application FastAPI
app = FastAPI()
# DONE : Ouvrir et charger en mémoire le pickle qui sérialise le model
with open("./src/titanic/api/resources/model.pkl", "rb") as f:
    model = pickle.load(f)
# DONE : Créer les class et dataclass représentant la donnée qui sera transmise au Webservice pour l'inférence
# DONE : Créer Pclass (enum)
# DONE : Créer Sex (enum)
# DONE : Créer Passenger (attention, l'objet doit pouvoir être transmis en dictionnaire au model. Il faudra créer une méthode d'instance
class Pclass(Enum):
    UPPER = 1
    MIDDLE = 2
    LOW = 3


class Sex(Enum):
    MALE = "male"
    FEMALE = "female"


@dataclass
class Passenger:
    pclass: Pclass
    sex: Sex
    sibSp: int
    parch: int

    def to_dict(self) -> dict:
        return {"Pclass": self.pclass.value, "Sex": self.sex.value, "SibSp": self.sibSp, "Parch": self.parch}
    
# TODO : Faire en sorte que cette fonction soit exposée via une toute GET /health
@app.get("/health")
def health() -> dict:
    return {"status": "OK"}

# DONE : Ajouter les paramètres de la fonction (peut se faire en deux fois avec la sécurisation via oAuth2)
@app.post("/infer")
def infer(passenger: Passenger, token: str = Depends(verify_token("api:read"))) -> list:

    df_passenger = pd.DataFrame([passenger.to_dict()])
    df_passenger["Sex"] = pd.Categorical(df_passenger["Sex"], categories=[Sex.FEMALE.value, Sex.MALE.value])
    df_to_predict = pd.get_dummies(df_passenger)

    res = model.predict(df_to_predict)

    return res.tolist()

