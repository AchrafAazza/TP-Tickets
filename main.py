
import pandas as pd
import re
from nltk.stem.snowball import FrenchStemmer
from simplemma import lemmatize

# ÉTAPE 1 : CHARGEMENT ET INSPECTION
df = pd.read_json("corpus_B.json")
df["text_raw"] = df["text_raw"].fillna("")

print("Nombre de tickets :", len(df))
print("Tickets vides :", df[df["text_raw"].str.strip() == ""]["id"].tolist())

# Clé de doublon avant masquage
df["cle_doublon"] = df["text_raw"].apply(
    lambda x: " ".join(x.lower().split())
)

# ÉTAPE 2 : MASQUAGE
def masquer(texte):
    texte = re.sub(r"https?://[^\s;]+", "URLTOKEN", texte, flags=re.I)
    texte = re.sub(r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}",
                   "EMAILTOKEN", texte)
    texte = re.sub(r"(\btel\s*:\s*)\d{10}\b",
                   r"\1PHONETOKEN", texte, flags=re.I)
    return texte.strip()

df["texte_masque"] = df["text_raw"].apply(masquer)

# ÉTAPE 3 : TOKENISATION ET FILTRAGE
motif = (
    r"EMAILTOKEN|PHONETOKEN|URLTOKEN|"
    r"[A-Za-z]+\d+[A-Za-z0-9]*|"
    r"\d+(?:[.:]\d+)+|\d+|[^\W\d_]+"
)

stopwords = {"le", "la", "les", "de", "du", "des",
             "à", "sur", "et", "un", "une", "l"}

def proteger(mot):
    return (
        mot.upper() in {"EMAILTOKEN", "PHONETOKEN", "URLTOKEN"}
        or bool(re.fullmatch(
            r"[A-Za-z]+\d+[A-Za-z0-9]*|\d+(?:[.:]\d+)*", mot
        ))
        or mot.lower() in {"python", "dh"}
    )

df["tokens"] = df["texte_masque"].apply(
    lambda x: re.findall(motif, x, flags=re.I)
)

df["formes"] = df["tokens"].apply(
    lambda mots: [m for m in mots
                  if proteger(m) or m.lower() not in stopwords]
)

# ÉTAPE 4 : STEMMING ET LEMMATISATION
stemmer = FrenchStemmer()

df["stems"] = df["formes"].apply(
    lambda mots: [m if proteger(m) else stemmer.stem(m.lower())
                  for m in mots]
)

df["lemmes"] = df["formes"].apply(
    lambda mots: [m if proteger(m) else lemmatize(m.lower(), lang="fr")
                  for m in mots]
)

# DÉTECTION DES DOUBLONS
df["doublons"] = df.groupby("cle_doublon")["id"].transform(
    lambda ids: ", ".join(ids) if len(ids) > 1 else ""
)

# RÉSULTATS
print("\n=== TEXTES MASQUÉS ===")
print(df[["id", "texte_masque"]].to_string(index=False))

print("\n=== DOUBLONS ===")
print(df[df["doublons"] != ""][["id", "doublons"]].to_string(index=False))

# EXPORT CSV
df.to_csv("resultats_B.csv", index=False, encoding="utf-8-sig")
print("\nRésultats enregistrés dans resultats_B.csv")
