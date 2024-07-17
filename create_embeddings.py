import pandas as pd
from dotenv import load_dotenv
from utils.embeddings_utils import get_embedding

load_dotenv()

# csv datei fragen_und_antworten.csv auslesen
# embedding für jede frage berechnen
# frage, antwort, embedding in csv datei speichern
input_datapath = "data/fragen_und_antworten.csv"
df = pd.read_csv(input_datapath, sep=';', on_bad_lines='skip')
df["embedding"] = df["Frage"].apply(lambda x: get_embedding(x))
df.to_csv("data/fragen_und_antworten_embeddings.csv", sep=';', index=False)
print("Data successfully processed and saved.")




