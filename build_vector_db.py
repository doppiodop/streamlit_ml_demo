from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import SentenceTransformerEmbeddings

from langchain.schema import Document
import pandas as pd

# Load recipes
df = pd.read_csv("13k-recipes.csv")

# Convert each recipe into a LangChain Document
docs = []
for _, row in df.iterrows():
    content = f"{row['Title']}\nIngredients: {row['Ingredients']}\nInstructions: {row['Instructions']}"
    docs.append(Document(page_content=content))

# Use SentenceTransformers for embeddings
embedding = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

# Create FAISS vector store
db = FAISS.from_documents(docs, embedding)

# Save it
db.save_local("vectorstore/recipes_faiss")
print("✅ Vector DB saved!")
