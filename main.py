from fastapi import FastAPI
from search import search_keywords;
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Ajouter le middleware CORS pour autoriser les requêtes depuis le frontend React
origins = [
    "http://localhost:5173",  # L'adresse du frontend React en développement (par défaut)
    "http://127.0.0.1:5173",  # Ajoutez cette ligne si vous utilisez localhost ou 127.0.0.1
    # Vous pouvez ajouter d'autres domaines ici
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # Liste des domaines autorisés
    allow_credentials=True,
    allow_methods=["*"],  # Permet toutes les méthodes (GET, POST, etc.)
    allow_headers=["*"],  # Permet tous les en-têtes
)
@app.get("/search")
def read_root(query: str):
    return search_keywords(query)
