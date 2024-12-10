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
        query_vector = [1 if word in keywords.split() else 0 for word in vectorizer]

        # Calculer les similarités cosinus
        tfidf_matrix = df.values
        similarities = cosine_similarity([query_vector], tfidf_matrix)

        # Afficher les résultats triés
        scores = list(zip(df.index, similarities[0]))
        # Filtrer uniquement les scores non nuls
        filtered_scores = [score for score in scores if score[1] > 0]
        sorted_scores = sorted(filtered_scores, key=lambda x: x[1], reverse=True)

        print("\nRésultats de recherche :")
        if sorted_scores:
            for file, score in sorted_scores:
                print(f"{file} : {score:.2f}")
        else:
            print("Aucun résultat trouvé avec ces mots-clés.")
    except Exception as e:
        print(f"Erreur lors de la recherche : {e}")

if __name__ == "__main__":
    query = input("Entrez les mots-clés à rechercher : ")
    search_keywords(query)
