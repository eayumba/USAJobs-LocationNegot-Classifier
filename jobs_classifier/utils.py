from typing import List
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


def tfidf_stacker(train_text: List[str], test_text: List[str]):
    word_vectorizer = TfidfVectorizer(
        sublinear_tf=True,
        stop_words="english",
        strip_accents="unicode",
        lowercase=True,
        analyzer="word",
        token_pattern=r"\w{1,}",
        ngram_range=(1, 3),
        dtype=np.float32,
        max_features=5000,
    )

    char_vectorizer = TfidfVectorizer(
        sublinear_tf=True,
        strip_accents="unicode",
        lowercase=True,
        analyzer="char",
        ngram_range=(1, 4),
        dtype=np.float32,
        max_features=2000,
    )

    word_vectorizer.fit(train_text)
    char_vectorizer.fit(train_text)
    word_feature_names = word_vectorizer.get_feature_names()
    char_feature_names = char_vectorizer.get_feature_names()

    all_feature_names = np.hstack([word_feature_names, char_feature_names])
    train_word_features = word_vectorizer.transform(train_text).toarray()
    train_char_features = char_vectorizer.transform(train_text).toarray()

    test_word_features = word_vectorizer.transform(test_text).toarray()
    test_char_features = char_vectorizer.transform(test_text).toarray()

    train_features = np.hstack([train_word_features, train_char_features])
    test_features = np.hstack([test_word_features, test_char_features])

    return train_features, test_features, all_feature_names
