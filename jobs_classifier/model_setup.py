import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC


from typing import List
import pickle


class tfidf_trainer:

    """
    Parameters:
        load: boolean
            If set to True, the preloaded weights will be read in from the pickle
            file. Users will then need to run the .transform method to output
            the tfidf vectors from the pretrained model

            If False, this means the users wants to retrain the tfidf model.
            In this scenario, the user should run both the fit and transform method
    """

    def __init__(self, pretrained_model: bool = False):

        self.pretrained_model = pretrained_model

        if self.pretrained_model:

            with open("pickle_files/word_vectorizer.pk", "rb") as f:
                self.preloaded_word_vectorizer = pickle.load(f)
            with open("pickle_files/char_vectorizer.pk", "rb") as f:
                self.preloaded_char_vectorizer = pickle.load(f)

    def fit(self, text_data: List[str]):
        self.new_word_vectorizer = TfidfVectorizer(
            sublinear_tf=True,
            stop_words="english",
            strip_accents="unicode",
            lowercase=True,
            analyzer="word",
            token_pattern=r"\w{1,}",
            ngram_range=(1, 3),
            dtype=np.float32,
            max_features=8000,
        )

        self.new_char_vectorizer = TfidfVectorizer(
            sublinear_tf=True,
            strip_accents="unicode",
            lowercase=True,
            analyzer="char_wb",
            ngram_range=(1, 4),
            dtype=np.float32,
            max_features=5000,
        )

        self.new_word_vectorizer.fit(text_data)
        self.new_char_vectorizer.fit(text_data)

        for fit_model in [
            (self.new_word_vectorizer, "word_vectorizer"),
            (self.new_char_vectorizer, "char_vectorizer"),
        ]:
            with open(f"pickle_files/{fit_model[1]}.pk", "wb") as fin:
                pickle.dump(fit_model[0], fin)

    def transform(self, text_data: List[str]):
        if self.pretrained_model:
            tfidf_word = self.preloaded_word_vectorizer.transform(text_data).toarray()
            tfidf_char = self.preloaded_char_vectorizer.transform(text_data).toarray()
            tfidf_out = np.hstack([tfidf_word, tfidf_char])

        else:
            tfidf_word = self.new_word_vectorizer.transform(text_data).toarray()
            tfidf_char = self.new_char_vectorizer.transform(text_data).toarray()
            tfidf_out = np.hstack([tfidf_word, tfidf_char])

        return tfidf_out


class JobSummaryClassifier:
    def __init__(self, load_pretrained):
        self.load_pretrained = load_pretrained
        if self.load_pretrained:
            with open("pickle_files/classifier_model.pk", "rb") as f:
                self.pretrained_model = pickle.load(f)

    def fit(self, X, y):
        self.model = SVC(kernel="linear", probability=True, random_state=1234)
        self.model.fit(X, y)
        with open("pickle_files/classifier_model.pk", "wb") as f:
            pickle.dump(self.model, f)
            print("model trained")

    def predict(self, X):
        if self.load_pretrained:
            prediction = self.pretrained_model.predict_proba(X)
        else:
            prediction = self.model.predict_proba(X)

        return prediction
