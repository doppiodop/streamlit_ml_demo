import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from transformers import pipeline
from PIL import Image
import torch
import torchvision.transforms as transforms
from torchvision import models
import shap
import seaborn as sns
from sentence_transformers import SentenceTransformer
import random
import torch

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

st.set_page_config(page_title="Cool ML Demos", layout="wide")

# 🎉 WELCOME SECTION
st.title("Welcome to Streamlit ML Demos!")
st.markdown("Explore several useful machine learning applications.")

# Dropdown with no selection by default
demo_options = [
    "-- Select a Demo --",
    # "Semantle",
    "Tabular Prediction",
    "Clustering Visualization",
    "Anomaly Detection",
    "Interpolator/Fitting Demo",
    "Recipe Assistant (RAG)"
]

option = st.selectbox("Choose a Demo", demo_options)

# Stop execution until user selects a demo
if option == "-- Select a Demo --":
    st.info("Please select a demo from the dropdown above to begin.")
    st.stop()


# # 2. SEMANTLE-STYLE WORD GAME
# if option == "Semantle":
#     st.header("🧩 Semantle-Style Word Similarity Game")

#     # 30 sample words
#     word_list = [
#         "cat", "dog", "apple", "city", "music", "river", "mountain", "car", "pencil", "computer",
#         "flower", "ocean", "book", "guitar", "house", "bicycle", "tree", "coffee", "sun", "moon",
#         "train", "phone", "painting", "bird", "school", "shoe", "bread", "forest", "movie", "star"
#     ]

#     @st.cache_resource
#     def get_model():
#         from sentence_transformers import SentenceTransformer
#         return SentenceTransformer('all-MiniLM-L6-v2')

#     model = get_model()

#     if "target_word" not in st.session_state:
#         st.session_state.target_word = random.choice(word_list)
#         st.session_state.guesses = []
#         st.session_state.similarities = []
#         st.session_state.revealed = False

#     st.markdown(
#         f"""
#         Guess the secret word! The target word is randomly chosen from a set of 30 possible English nouns.<br>
#         You can guess **any word** in English. For each guess, you'll see how "close" your guess is to the target word based on semantic similarity.
#         """, unsafe_allow_html=True
#     )

#     guess = st.text_input("Your guess:", "")

#     # Main guess logic
#     if st.button("Submit Guess") and not st.session_state.revealed:
#         if guess.strip():
#             emb_guess = model.encode([guess])[0]
#             emb_target = model.encode([st.session_state.target_word])[0]
#             sim = float(torch.nn.functional.cosine_similarity(
#                 torch.tensor(emb_guess), torch.tensor(emb_target), dim=0
#             ).item())
#             st.session_state.guesses.append(guess.lower())
#             st.session_state.similarities.append(sim)
#             if guess.lower() == st.session_state.target_word:
#                 st.success(f"🎉 Correct! The target word was **{st.session_state.target_word}**.")
#                 st.balloons()
#                 st.session_state.revealed = True
#             else:
#                 st.info(f"Similarity to target: **{sim:.3f}** (1.0 is identical, -1.0 is opposite)")

#     # Reveal Target Word Button
#     if st.button("Reveal Target Word (Give Up)"):
#         st.session_state.revealed = True

#     # Show the target word if revealed or solved
#     if st.session_state.revealed:
#         st.warning(f"The target word was: **{st.session_state.target_word}**")
#         if st.button("Start New Game"):
#             st.session_state.target_word = random.choice(word_list)
#             st.session_state.guesses = []
#             st.session_state.similarities = []
#             st.session_state.revealed = False
#         st.stop()

#     # Show ranking of guesses
#     if st.session_state.guesses:
#         st.markdown("### Guesses Ranking (most similar first):")
#         df = pd.DataFrame({
#             "Word": st.session_state.guesses,
#             "Similarity": st.session_state.similarities
#         })
#         df = df.sort_values("Similarity", ascending=False).reset_index(drop=True)
#         st.table(df)
#         st.markdown(f"Target word is one of: {', '.join(word_list)}")

#     if st.button("Reset Game"):
#         st.session_state.target_word = random.choice(word_list)
#         st.session_state.guesses = []
#         st.session_state.similarities = []
#         st.session_state.revealed = False
# 3. TABULAR PREDICTION with SHAP, Confusion Matrix, ROC Curve
elif option == "Tabular Prediction":
    st.header("Tabular Prediction with RandomForest (Iris Dataset)")

    from sklearn.datasets import load_iris
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    import shap
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt

    # Load dataset
    iris = load_iris()
    X = pd.DataFrame(iris.data, columns=iris.feature_names)
    y = pd.Series(iris.target, name='target')
    df = X.copy()
    df['target'] = y
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

    # Display dataset preview
    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    # Display descriptive statistics
    st.subheader("Descriptive Statistics")
    st.dataframe(df.describe().T)

    # Model training
    model = RandomForestClassifier()
    model.fit(X_train, y_train)

    # Input sliders
    st.subheader("Enter Flower Features:")
    sepal_length = st.slider("Sepal Length", float(X.iloc[:, 0].min()), float(X.iloc[:, 0].max()), float(X.iloc[:, 0].mean()))
    sepal_width = st.slider("Sepal Width", float(X.iloc[:, 1].min()), float(X.iloc[:, 1].max()), float(X.iloc[:, 1].mean()))
    petal_length = st.slider("Petal Length", float(X.iloc[:, 2].min()), float(X.iloc[:, 2].max()), float(X.iloc[:, 2].mean()))
    petal_width = st.slider("Petal Width", float(X.iloc[:, 3].min()), float(X.iloc[:, 3].max()), float(X.iloc[:, 3].mean()))
    input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])

    # Prediction
    prediction = model.predict(input_data)[0]
    prediction_proba = model.predict_proba(input_data)

    st.success(f"**Predicted Class:** {iris.target_names[prediction]}")
    st.bar_chart(prediction_proba[0])

    # SHAP Explainability
    st.subheader("SHAP Explanation for this Prediction")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(input_data)

    # SHAP summary plot
    fig, ax = plt.subplots()
    shap.summary_plot(shap_values, input_data, feature_names=iris.feature_names, plot_type="bar", show=False)
    st.pyplot(fig)
    plt.close(fig)

    # SHAP waterfall plot
    try:
        if isinstance(shap_values, list) and len(shap_values) > 1:
            values = shap_values[prediction][0]
            base = explainer.expected_value[prediction]
        else:
            values = shap_values[0]
            base = explainer.expected_value

        fig2, ax2 = plt.subplots()
        shap.waterfall_plot(
            shap.Explanation(
                values=values,
                base_values=base,
                data=input_data[0],
                feature_names=iris.feature_names
            ),
            show=False
        )
        st.pyplot(fig2)
        plt.close(fig2)
    except Exception as e:
        st.warning(f"SHAP explanation not available: {e}")


# 4. LIVE DATA VISUALIZATION
elif option == "Clustering Visualization":
    st.header("Clustering Algorithms Comparison")

    import sklearn.datasets
    from sklearn.cluster import KMeans, AgglomerativeClustering, SpectralClustering, DBSCAN
    from sklearn.metrics import silhouette_score
    from scipy.spatial.distance import cdist

    # Demo datasets
    demo_datasets = {
        "Iris (first 2 features)": load_iris(return_X_y=True),
        "Simulated Circles": lambda: tuple(sklearn.datasets.make_circles(n_samples=300, factor=0.5, noise=0.05)),
        "Simulated Blobs": lambda: tuple(sklearn.datasets.make_blobs(n_samples=300, centers=4, n_features=2, random_state=42))
    }

    # Dataset selection
    dataset_name = st.selectbox("Choose a dataset", list(demo_datasets.keys()))
    if isinstance(demo_datasets[dataset_name], tuple):
        X, y_true = demo_datasets[dataset_name]
    else:
        X, y_true = demo_datasets[dataset_name]()

    # Only take first two features for plotting
    X_plot = X if X.shape[1] == 2 else X[:, :2]

    st.markdown("Or upload your own CSV (at least 2 numeric columns):")
    uploaded = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded:
        df = pd.read_csv(uploaded)
        numeric_cols = df.select_dtypes(include=np.number).columns
        if len(numeric_cols) < 2:
            st.error("The uploaded CSV must have at least two numeric columns.")
            st.stop()
        X_plot = df[numeric_cols[:2]].values
        st.write("Using columns:", numeric_cols[:2])
        y_true = None

    # Algorithm selection
    algo = st.selectbox("Clustering Algorithm", [
        "KMeans",
        "AgglomerativeClustering",
        "SpectralClustering",
        "DBSCAN"
    ])

    # K-based algorithms and their cluster selectors
    k_based = ["KMeans", "AgglomerativeClustering", "SpectralClustering"]
    if algo in k_based:
        k = st.slider("Number of clusters (k)", 2, 8, 3)
    if algo == "DBSCAN":
        eps = st.slider("DBSCAN: eps (distance threshold)", 0.05, 1.0, 0.2)
        min_samples = st.slider("DBSCAN: min_samples", 2, 20, 5)

    # Fit and predict
    if algo == "KMeans":
        clusterer = KMeans(n_clusters=k, random_state=0)
        y_pred = clusterer.fit_predict(X_plot)
        centers = clusterer.cluster_centers_
    elif algo == "AgglomerativeClustering":
        clusterer = AgglomerativeClustering(n_clusters=k)
        y_pred = clusterer.fit_predict(X_plot)
        centers = None
    elif algo == "SpectralClustering":
        clusterer = SpectralClustering(n_clusters=k, affinity='nearest_neighbors', assign_labels='kmeans', random_state=0)
        y_pred = clusterer.fit_predict(X_plot)
        centers = None
    elif algo == "DBSCAN":
        clusterer = DBSCAN(eps=eps, min_samples=min_samples)
        y_pred = clusterer.fit_predict(X_plot)
        centers = None

    # Plot clusters
    fig, ax = plt.subplots(figsize=(6,4))
    scatter = ax.scatter(X_plot[:, 0], X_plot[:, 1], c=y_pred, cmap='viridis', s=50, alpha=0.7)
    if centers is not None:
        ax.scatter(centers[:, 0], centers[:, 1], c="red", marker="X", s=200, label="Centers")
    ax.set_xlabel("Feature 1")
    ax.set_ylabel("Feature 2")
    ax.set_title(f"{algo} Clustering")
    if centers is not None:
        ax.legend()
    st.pyplot(fig)
    plt.close(fig)

    # Silhouette Score
    if len(set(y_pred)) > 1 and -1 not in set(y_pred):  # -1 is for noise in DBSCAN
        score = silhouette_score(X_plot, y_pred)
        st.markdown(f"Silhouette Score: **{score:.2f}**")
    elif algo == "DBSCAN" and len(set(y_pred)) > 1:
        # Exclude noise points (label -1) for silhouette
        mask = y_pred != -1
        if np.any(mask):
            score = silhouette_score(X_plot[mask], y_pred[mask])
            st.markdown(f"Silhouette Score (excluding noise): **{score:.2f}**")
        else:
            st.markdown("Silhouette Score: Not available (all points are noise)")
    else:
        st.markdown("Silhouette Score: Not available (only one cluster found)")

    # Elbow Plot for KMeans-like algorithms
    if algo == "KMeans":
        st.markdown("#### KMeans Elbow Plot")
        sse = []
        K_range = range(2, 11)
        for k_val in K_range:
            kmeans = KMeans(n_clusters=k_val, random_state=0)
            kmeans.fit(X_plot)
            sse.append(kmeans.inertia_)
        fig2, ax2 = plt.subplots()
        ax2.plot(K_range, sse, marker='o')
        ax2.set_xlabel("k")
        ax2.set_ylabel("Sum of Squared Errors (SSE)")
        ax2.set_title("Elbow Plot")
        st.pyplot(fig2)
        plt.close(fig2)

    elif algo in ["AgglomerativeClustering", "SpectralClustering"]:
        st.markdown("#### Silhouette Score by Number of Clusters")
        sil_scores = []
        K_range = range(2, 11)
        for k_val in K_range:
            if algo == "AgglomerativeClustering":
                clust = AgglomerativeClustering(n_clusters=k_val)
            else:
                clust = SpectralClustering(n_clusters=k_val, affinity='nearest_neighbors', assign_labels='kmeans', random_state=0)
            labels = clust.fit_predict(X_plot)
            if len(set(labels)) > 1:
                sil = silhouette_score(X_plot, labels)
            else:
                sil = np.nan
            sil_scores.append(sil)
        fig3, ax3 = plt.subplots()
        ax3.plot(K_range, sil_scores, marker='o')
        ax3.set_xlabel("Number of clusters")
        ax3.set_ylabel("Silhouette Score")
        ax3.set_title("Silhouette Score vs Number of Clusters")
        st.pyplot(fig3)
        plt.close(fig3)

elif option == "Anomaly Detection":
    st.header("Anomaly Detection Demo")

    import sklearn.datasets
    from sklearn.ensemble import IsolationForest
    from sklearn.neighbors import LocalOutlierFactor
    from sklearn.svm import OneClassSVM

    demo_datasets = {
        "Iris (first 2 features)": load_iris(return_X_y=True),
        "Simulated Blobs": lambda: tuple(sklearn.datasets.make_blobs(n_samples=300, centers=3, n_features=2, random_state=42)),
        "Simulated Moons": lambda: tuple(sklearn.datasets.make_moons(n_samples=300, noise=0.05, random_state=42))
    }

    # Dataset selection
    dataset_name = st.selectbox("Choose a dataset", list(demo_datasets.keys()))
    if isinstance(demo_datasets[dataset_name], tuple):
        X, y_true = demo_datasets[dataset_name]
    else:
        X, y_true = demo_datasets[dataset_name]()

    X_plot = X if X.shape[1] == 2 else X[:, :2]

    st.markdown("Or upload your own CSV (at least 2 numeric columns):")
    uploaded = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded:
        df = pd.read_csv(uploaded)
        numeric_cols = df.select_dtypes(include=np.number).columns
        if len(numeric_cols) < 2:
            st.error("The uploaded CSV must have at least two numeric columns.")
            st.stop()
        X_plot = df[numeric_cols[:2]].values
        st.write("Using columns:", numeric_cols[:2])
        y_true = None

    algo = st.selectbox("Algorithm", [
        "IsolationForest",
        "LocalOutlierFactor",
        "OneClassSVM"
    ])

    contamination = st.slider("Expected proportion of anomalies (contamination)", 0.01, 0.20, 0.05, step=0.01)

    # Fit and predict
    if algo == "IsolationForest":
        clf = IsolationForest(contamination=contamination, random_state=42)
        y_pred = clf.fit_predict(X_plot)
    elif algo == "LocalOutlierFactor":
        clf = LocalOutlierFactor(n_neighbors=20, contamination=contamination)
        y_pred = clf.fit_predict(X_plot)
    elif algo == "OneClassSVM":
        clf = OneClassSVM(nu=contamination, kernel="rbf", gamma='auto')
        y_pred = clf.fit_predict(X_plot)

    # -1 = anomaly, 1 = normal
    anomalies = y_pred == -1
    normals = y_pred == 1

    st.markdown(f"**Detected {np.sum(anomalies)} anomalies out of {len(X_plot)} samples.**")

    # Plot
    fig, ax = plt.subplots(figsize=(6,4))
    ax.scatter(X_plot[normals, 0], X_plot[normals, 1], c='blue', label="Normal", alpha=0.7)
    ax.scatter(X_plot[anomalies, 0], X_plot[anomalies, 1], c='red', label="Anomaly", marker='x', s=80)
    ax.set_xlabel("Feature 1")
    ax.set_ylabel("Feature 2")
    ax.set_title(f"Anomaly Detection ({algo})")
    ax.legend()
    st.pyplot(fig)
    plt.close(fig)

    # Show table of anomalies
    if np.sum(anomalies) > 0:
        st.markdown("### Anomalous Points")
        df_out = pd.DataFrame(X_plot[anomalies], columns=["Feature 1", "Feature 2"])
        st.dataframe(df_out)
        st.download_button("Download anomalies as CSV", df_out.to_csv(index=False), file_name="anomalies.csv")

    import spacy
    import networkx as nx
    from pyvis.network import Network
    import streamlit.components.v1 as components

    # Load spaCy model (cache for speed)
    @st.cache_resource
    def get_spacy():
        return spacy.load("en_core_web_sm")
    nlp = get_spacy()

elif option == "Interpolator/Fitting Demo":
    st.header("Interpolator & Curve Fitting Demo")

    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.optimize import curve_fit
    from numpy.polynomial import Polynomial
    import pandas as pd

    # Data generation/upload (fixed across reruns)
    if 'interp_x' not in st.session_state or 'interp_y' not in st.session_state:
        x = np.linspace(0, 10, 25)
        y = np.sin(x) + np.random.normal(0, 0.2, size=x.size)
        st.session_state.interp_x = x
        st.session_state.interp_y = y

    st.markdown("Use the demo data (noisy sine wave), or upload your own CSV (with columns x and y).")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Regenerate Demo Data"):
            x = np.linspace(0, 10, 25)
            y = np.sin(x) + np.random.normal(0, 0.2, size=x.size)
            st.session_state.interp_x = x
            st.session_state.interp_y = y

    with col2:
        uploaded = st.file_uploader("Upload CSV (x and y columns)", type=["csv"])
        if uploaded:
            df = pd.read_csv(uploaded)
            if "x" in df.columns and "y" in df.columns:
                st.session_state.interp_x = df["x"].values
                st.session_state.interp_y = df["y"].values
            else:
                st.error("CSV must have columns named 'x' and 'y'.")
                st.stop()

    x = st.session_state.interp_x
    y = st.session_state.interp_y

    fit_type = st.selectbox(
        "Choose fit function",
        ["Linear", "Polynomial", "Exponential", "Logarithmic", "Spline"]
    )
    if fit_type == "Polynomial":
        deg = st.slider("Polynomial degree", 2, 8, 3)

    def linear(x, a, b): return a * x + b
    def exponential(x, a, b): return a * np.exp(b * x)
    def logarithmic(x, a, b): return a * np.log(x + 1e-8) + b  # avoid log(0)

    fit_x = np.linspace(x.min(), x.max(), 400)
    fit_y = None
    fit_label = ""
    r2 = None

    try:
        if fit_type == "Linear":
            popt, _ = curve_fit(linear, x, y)
            fit_y = linear(fit_x, *popt)
            fit_label = f"Linear Fit: y={popt[0]:.2f}x+{popt[1]:.2f}"
            y_pred = linear(x, *popt)
        elif fit_type == "Polynomial":
            coefs = np.polyfit(x, y, deg)
            poly = np.poly1d(coefs)
            fit_y = poly(fit_x)
            fit_label = f"Degree {deg} Polynomial"
            y_pred = poly(x)
        elif fit_type == "Exponential":
            popt, _ = curve_fit(exponential, x, y, maxfev=10000)
            fit_y = exponential(fit_x, *popt)
            fit_label = f"Exp Fit: y={popt[0]:.2f}exp({popt[1]:.2f}x)"
            y_pred = exponential(x, *popt)
        elif fit_type == "Logarithmic":
            popt, _ = curve_fit(logarithmic, x, y)
            fit_y = logarithmic(fit_x, *popt)
            fit_label = f"Log Fit: y={popt[0]:.2f}log(x)+{popt[1]:.2f}"
            y_pred = logarithmic(x, *popt)
        elif fit_type == "Spline":
            from scipy.interpolate import UnivariateSpline
            spline = UnivariateSpline(x, y, s=1)
            fit_y = spline(fit_x)
            fit_label = "Spline Fit"
            y_pred = spline(x)
        # Compute R^2
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r2 = 1 - (ss_res / ss_tot)
    except Exception as e:
        st.warning(f"Could not fit selected function: {e}")

    fig, ax = plt.subplots()
    ax.scatter(x, y, label="Data", color="black")
    if fit_y is not None:
        ax.plot(fit_x, fit_y, label=f"{fit_label} (R²={r2:.2f})")
    ax.legend()
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    st.pyplot(fig)
    plt.close(fig)
# --- RECIPE ASSISTANT (RAG) Demo ---
elif option == "Recipe Assistant (RAG)":
    st.header("Recipe Assistant with LangChain + Groq + FAISS")

    @st.cache_resource
    def load_embedding_and_db():
        embedding = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
        db = FAISS.load_local(
            "vectorstore/recipes_faiss",
            embedding,
            allow_dangerous_deserialization=True
        )
        return embedding, db

    embedding, db = load_embedding_and_db()

    @st.cache_resource
    def load_llm():
        return ChatGroq(
            temperature=0.3,
            model_name="llama-3.3-70b-versatile",
            api_key=os.getenv("GROQ_API_KEY")
        )

    llm = load_llm()

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

    @st.cache_resource
    def load_qa_chain():
        return RetrievalQA.from_chain_type(
            llm=llm,
            retriever=db.as_retriever(search_type="similarity", k=3),
            chain_type_kwargs={"prompt": prompt}
        )

    qa_chain: Runnable = load_qa_chain()
    retriever = db.as_retriever(search_type="similarity", k=3)

    # Session state for chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_question = st.chat_input("What do you want to cook or ask?")

    if user_question:
        # Save user message
        st.session_state.messages.append({"role": "user", "content": user_question})

        # Show user message
        with st.chat_message("user"):
            st.markdown(user_question)

        # Retrieve recipes and display snippet in expander
        with st.spinner("Retrieving matching recipes..."):
            retrieved_docs = retriever.get_relevant_documents(user_question)

        with st.expander("Retrieved Recipes", expanded=False):
            for i, doc in enumerate(retrieved_docs):
                st.markdown(f"**Recipe {i+1}:**\n```\n{doc.page_content}\n```")

        # Generate answer from LLM
        with st.spinner("Thinking..."):
            answer = qa_chain.run(user_question)

        # Save assistant response
        st.session_state.messages.append({"role": "assistant", "content": answer})

        # Show assistant message
        with st.chat_message("assistant"):
            st.markdown(answer)
