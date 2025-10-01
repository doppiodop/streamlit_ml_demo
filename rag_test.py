import os
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import Runnable

# Load environment variables
load_dotenv()

# Set up embeddings and load FAISS vector store
embedding = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
db = FAISS.load_local(
    "vectorstore/recipes_faiss",
    embedding,
    allow_dangerous_deserialization=True
)


# Initialize the LLM with Groq
llm = ChatGroq(
    temperature=0.3,
    model_name="llama-3.3-70b-versatile",  # You can change this if needed
    api_key=os.getenv("GROQ_API_KEY")
)

# Define the custom prompt
prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""
You are a helpful cooking assistant.

Based on the following recipe information:
{context}

Answer the user's question or suggest a recipe:
Question: {question}

Return always 3 recipes if i tell you the ingredients i have, however if i'm specific about a recipe, just tell me about the recipe i asked you
"""
)

# Set up the RAG chain
qa_chain: Runnable = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=db.as_retriever(search_type="similarity", k=3),
    chain_type_kwargs={"prompt": prompt}
)



def main():
    print("👨‍🍳 Welcome to the Recipe Assistant!")
    
    # Get retriever once, outside the loop
    retriever = db.as_retriever(search_type="similarity", k=3)
    
    while True:
        question = input("\nWhat do you want to cook or ask? (type 'exit' to quit): ")
        if question.lower() in {"exit", "quit"}:
            break
        
        print("\n🔍 Retrieving matching recipes...\n")
        retrieved_docs = retriever.get_relevant_documents(question)
        
        print("📚 Retrieved Recipes:")
        for i, doc in enumerate(retrieved_docs):
            print(f"\n--- Recipe {i+1} ---")
            print(doc.page_content)
        
        print("\n🤖 Thinking...\n")
        answer = qa_chain.run(question)
        
        print("\n🍽️ Suggested Recipe / Info:\n")
        print(answer)


if __name__ == "__main__":
    main()
