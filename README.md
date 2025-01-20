
# NLP Assignment 1 (AIT - DSAI)

- [Student Information](#student-information)
- [Files Structure](#files-structure)
- [How to run](#how-to-run)
- [Dataset](#dataset)
- [Evaluation](#evaluation)

## Student Information
 - Name: Soe Htet Naing
 - ID: st125166

## Files Structure
 - The `code` folder contains Jupyter Notebook files for training and testing different models:
   - Skipgram
   - Skipgram with Negative Sampling
   - GloVe
   - Gensim
 - The `app` folder contains:
   - `app.py`: Main web application to run locally.
   - The `models` folder: Contains models, args and `classes.py`.
 - `samples_images` folder includes sample screenshots of the web application with sample queries for all models.
 - Dataset files:
   - `word-testsemantic.v1.txt`: Semantic data (e.g., capital-common-countries).
   - `word-testsyntatic.v1.txt`: Syntactic data (e.g., past-tense).
   - `wordsim_similarity_goldstandard.txt`: Similarity dataset.

## How to run
1. Ensure all dependencies are installed.
2. Navigate to the `app` folder.
3. Run `streamlit run app.py`.
4. Open `http://localhost:8502` in your browser.

## Dataset
- Training data used: `brown` dataset (category: `news`) from `nltk`.
- Parameters:
  - Batch size: 2
  - Embedding dimension: 2
  - Window size: 2
  - Epochs: 100

## Evaluation

| Model             | Training Loss | Training Time   | Semantic Accuracy | Syntactic Accuracy | Similarity Score |
|-------------------|---------------|-----------------|--------------------|---------------------|------------------|
| Skipgram          | 10.73         | 0 min 42 sec    | 0.00%             | 0.00%              | -0.02            |
| Skipgram (NEG)    | 1.85          | 0 min 37 sec    | 0.00%             | 0.00%              | 0.09             |
| Glove             | 3.79          | 0 min 05 sec    | 0.00%             | 0.00%              | 0.06             |
| Glove (Gensim)    | -             | -               | 53.16%            | 55.45%             | 0.54             |

