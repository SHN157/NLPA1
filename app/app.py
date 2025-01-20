import streamlit as st
import numpy as np
import pickle
import torch
from models.classes import Skipgram, SkipgramNeg, Glove, Gensim
import nltk
from nltk.corpus import brown

nltk.download('brown')
corpus = brown.sents(categories="news")

# Load Models
@st.cache_resource
def load_models():
    # Skipgram
    skg_args = pickle.load(open('models/skipgram.args', 'rb'))
    model_skipgram = Skipgram(**skg_args)
    model_skipgram.load_state_dict(torch.load('models/skipgram.model'))

    # SkipgramNeg
    neg_args = pickle.load(open('models/skipgramneg.args', 'rb'))
    model_neg = SkipgramNeg(**neg_args)
    model_neg.load_state_dict(torch.load('models/skipgramneg.model'))

    # Glove
    glove_args = pickle.load(open('models/glovefromscratch.args', 'rb'))
    model_glove = Glove(**glove_args)
    model_glove.load_state_dict(torch.load('models/glovefromscratch.model'))

    # Gensim
    load_model = pickle.load(open('models/gensim.model', 'rb'))
    model_gensim = Gensim(load_model)

    return {
        'skipgram': model_skipgram,
        'neg': model_neg,
        'glove': model_glove,
        'gensim': model_gensim,
    }

models = load_models()
model_names = {'skipgram': 'Skipgram', 'neg': 'SkipgramNeg', 'glove': 'Glove', 'gensim': 'Glove (Gensim)'}

# Helper function for finding similar sentences
def find_closest_indices_cosine(vector_list, single_vector, k=10):
    similarities = np.dot(vector_list, single_vector) / (
        np.linalg.norm(vector_list, axis=1) * np.linalg.norm(single_vector)
    )
    top_indices = np.argsort(similarities)[-k:][::-1]
    return top_indices

# App UI
st.title("NLP A1: Search Engine")

# Sidebar for model selection
st.sidebar.title("Select NLP Model")
selected_model = st.sidebar.selectbox("Choose a model:", list(model_names.keys()), format_func=lambda x: model_names[x])

# Query input
query = st.text_input("Enter your search query:", placeholder="Type your query here.")

# Submit button
if st.button("Search"):
    if query:
        model = models[selected_model]

        # Compute sentence embedding for the query
        qwords = query.split()
        qwords_embeds = np.array([model.get_embed(word) for word in qwords])
        qsentence_embeds = np.mean(qwords_embeds, axis=0)

        # Compute embeddings for the corpus
        corpus_embeds = []
        for each_sent in corpus:
            words_embeds = np.array([model.get_embed(word) for word in each_sent])
            sentence_embeds = np.mean(words_embeds, axis=0)
            corpus_embeds.append(sentence_embeds)

        corpus_embeds = np.array(corpus_embeds)

        # Find the closest sentences
        result_idxs = find_closest_indices_cosine(corpus_embeds, qsentence_embeds)
        results = [' '.join(corpus[idx]) for idx in result_idxs]

        # Display results
        st.subheader(f"Search Results using {model_names[selected_model]}:")
        for idx, sentence in enumerate(results, 1):
            st.write(f"{idx}. {sentence}")
    else:
        st.warning("Please enter a query.")
