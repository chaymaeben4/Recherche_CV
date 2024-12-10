import pdfplumber
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import os
from whoosh.index import create_in
from whoosh.fields import Schema, TEXT
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import string
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
from collections import defaultdict
import json


# Télécharger les ressources nécessaires pour nltk
nltk.download('stopwords')

# Initialisation de Porter Stemmer
stemmer = PorterStemmer()

# Liste étendue des mots vides à exclure
custom_stopwords = set(stopwords.words('english')) | {
     "i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours", "yourself", "yourselves",
                "he", "him", "his", "himself", "she", "her", "hers", "herself", "it", "its", "itself", "they", "them", "their",
                "theirs", "themselves", "what", "which", "who", "whom", "this", "that", "these", "those", "am", "is", "are",
                "was", "were", "be", "been", "being", "have", "has", "had", "having", "do", "does", "did", "doing", "a", "an",
                "the", "and", "but", "if", "or", "because", "as", "until", "while", "of", "at", "by", "for", "with", "about",
                "against", "between", "into", "through", "during", "before", "after", "above", "below", "to", "from", "up",
                "down", "in", "out", "on", "off", "over", "under", "again", "further", "then", "once", "here", "there",
                "when", "where", "why", "how", "all", "any", "both", "each", "few", "more", "most", "other", "some",
                "such", "no", "nor", "not", "only", "own", "same", "so", "than", "too", "very", "s", "t", "can", "will",
                "just", "don", "should", "now", "d", "ll", "m", "o", "re", "ve", "y", "ain", "aren", "couldn", "didn",
                "doesn", "hadn", "hasn", "haven", "isn", "ma", "mightn", "mustn", "needn", "shan", "shouldn", "wasn", "weren",
                "won", "wouldn"
}

# Fonction pour extraire le texte d'un PDF
def extract_text_with_pdfplumber(pdf_path):
    extracted_text = ""
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                extracted_text += page.extract_text()
    except Exception as e:
        print(f"Erreur lors de l'extraction : {e}")
    return extracted_text

# Fonction de nettoyage du texte
def clean_extracted_text(text):
    # Supprimer les lignes vides et normaliser les espaces
    cleaned_text = "\n".join([line.strip() for line in text.splitlines() if line.strip()])
    return cleaned_text

# Fonction de normalisation des termes extraits avec Porter Stemmer
def normalize_text_with_stemming(text):
    words = text.split()
    stemmed_words = []
    for word in words:
        # Supprimer la ponctuation autour des mots
        word = word.strip(string.punctuation)
        # Mettre en minuscule et appliquer le stemming
        stemmed_word = stemmer.stem(word.lower())
        # Filtrer les mots : non-stopwords, alphabétiques et significatifs
        if (
            stemmed_word not in custom_stopwords  # Pas un mot vide
            and len(stemmed_word) > 2            # Longueur significative
            and stemmed_word.isalpha()           # Contient uniquement des lettres
        ):
            stemmed_words.append(stemmed_word)
    return stemmed_words

# Définir le schéma pour l'index
schema = Schema(title=TEXT(stored=True), content=TEXT(stored=True))

# Créer l'index dans un dossier spécifique
if not os.path.exists("index"):
    os.mkdir("index")

ix = create_in("index", schema)

# Fonction pour ajouter un CV à l'index
def add_cv_to_index(pdf_path, index, index_inverse):
    # Extraire le texte du fichier PDF
    text = extract_text_with_pdfplumber(pdf_path)
    text1 = clean_extracted_text(text)
    
    # Normaliser les termes extraits avec Porter Stemmer
    stemmed_terms = normalize_text_with_stemming(text1)
    text2 = " ".join(stemmed_terms)  # Joindre les termes normalisés

    # Ajouter le document à l'index Whoosh
    writer = index.writer()
    writer.add_document(title=pdf_path, content=text2)
    writer.commit()

    # Récupérer uniquement le nom du fichier
    filename = os.path.basename(pdf_path)

    # Mettre à jour l'index inversé
    for term in stemmed_terms:
        if term not in index_inverse:
            index_inverse[term] = []
        # Ajouter le fichier uniquement s'il n'est pas déjà présent
        if filename not in index_inverse[term]:
            index_inverse[term].append(filename)


# Fonction mise à jour pour générer un fichier JSON avec les index inversés
def generate_json_with_index(json_filename, index_inverse):
    try:
        with open(json_filename, 'w', encoding='utf-8') as f:
            json.dump(index_inverse, f, ensure_ascii=False, indent=4)
        print(f"Index inversé sauvegardé dans : {json_filename}")
    except Exception as e:
        print(f"Erreur lors de la génération du fichier JSON : {e}")


#Ponderation
def calculate_tfidf_from_pdfs(pdf_paths):
    corpus = []
    filenames = []

    # Préparer les documents normalisés
    for pdf_path in pdf_paths:
        text = extract_text_with_pdfplumber(pdf_path)
        normalized_terms = normalize_text_with_stemming(text)
        corpus.append(" ".join(normalized_terms))
        filenames.append(os.path.basename(pdf_path))

    # Calculer les pondérations TF-IDF
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(corpus)
    feature_names = vectorizer.get_feature_names_out()

    # Convertir la matrice TF-IDF en DataFrame
    tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), index=filenames, columns=feature_names)

    return tfidf_df

# Fonction pour extraire les termes les plus importants
def extract_top_terms(tfidf_df, top_n=5):
    top_terms = defaultdict(list)

    for filename, row in tfidf_df.iterrows():
        # Trier les termes par poids TF-IDF décroissant
        sorted_terms = row.sort_values(ascending=False).head(top_n)
        top_terms[filename] = sorted_terms.index.tolist()

    return top_terms

# Fonction principale
def process_and_weight_pdfs(pdf_paths):
    # Calculer la matrice TF-IDF
    tfidf_df = calculate_tfidf_from_pdfs(pdf_paths)

    # Extraire les termes les plus importants
    top_terms = extract_top_terms(tfidf_df, top_n=5)

    # Exporter les pondérations TF-IDF vers un CSV
    tfidf_df.to_csv("tfidf_results.csv", index=True)
    print("Les pondérations TF-IDF ont été exportées vers 'tfidf_results.csv'.")

    return tfidf_df, top_terms




# Exemple d'utilisation
index_inverse = {}
cv_folder = "./Cvs"
pdf_paths = [
    os.path.join(cv_folder, file)
    for file in os.listdir(cv_folder)
    if file.endswith(".pdf")  # Filtrer uniquement les fichiers PDF
]







# Ajouter chaque CV à l'index et mettre à jour l'index inversé
for pdf_path in pdf_paths:
    add_cv_to_index(pdf_path, ix, index_inverse)

# Générer le fichier JSON avec les index inversés
output_json = "index_inverse.json"
generate_json_with_index(output_json, index_inverse)

# Générer le PDF avec le tableau des index
#output_pdf = "index_table.pdf"
#generate_pdf_with_index(output_pdf, index_inverse)
#print(f"PDF généré: {output_pdf}")

# Lancer le traitement
tfidf_results, top_terms_per_doc = process_and_weight_pdfs(pdf_paths)

# Afficher les termes les plus importants par document
print("Termes les plus importants par document :")
for doc, terms in top_terms_per_doc.items():
    print(f"{doc}: {', '.join(terms)}")