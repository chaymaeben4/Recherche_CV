import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

INDEX_FILE = "tfidf_results.csv"

# Fonction pour rechercher les mots-clés
def search_keywords(keywords):
    try:
        # Charger les données TF-IDF
        df = pd.read_csv(INDEX_FILE, index_col=0)

        # Convertir les mots-clés en vecteur
        vectorizer = df.columns
        keywords_list = keywords.split(' ')
        query_vector = [1 if word in keywords_list else 0 for word in vectorizer]

        # Calculer les similarités cosinus
        tfidf_matrix = df.values
        similarities = cosine_similarity([query_vector], tfidf_matrix)

        # Préparer les résultats avec les mots-clés présents
        results = []
        for idx, (file_name, similarity) in enumerate(zip(df.index, similarities[0])):
            # Liste des mots présents dans ce fichier
            present_keywords = [word for word in keywords_list if word in df.columns[df.iloc[idx] > 0]]
            
            if present_keywords:  # Filtrer les fichiers sans mots-clés
                results.append((file_name, present_keywords, float(similarity)))  # Convertir en float standard

        # Trier les résultats :
        # D'abord par nombre de mots-clés présents (prioriser les fichiers avec les 2 mots),
        # puis par similarité.
        sorted_results = sorted(results, key=lambda x: (-len(x[1]), -x[2]))

        # Retourner les résultats triés
        return sorted_results if sorted_results else [("Aucun résultat", [], 0.0)]

    except Exception as e:
        return [f"Erreur lors de la recherche : {e}"]

# Exemple d'appel à la fonction
# results = search_keywords(" html java ")
# print(results)