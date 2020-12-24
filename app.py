import json
import os

import emoji
import numpy as np
import pandas as pd
import streamlit as st

from torchmoji.sentence_tokenizer import SentenceTokenizer
from torchmoji.model_def import torchmoji_emojis

from constants import (
    EMOJIS, REVIEWS_DATA_PATH,
    PRETRAINED_PATH, VOCAB_PATH,
    GAME_LIST, REVIEWS,  # TODO delete in the final version, only for dev branch
)


@st.cache
def load_data():
    columns = ["name", "reviewer_name", "review_rating", "review_title", "review_content"]
    with open(REVIEWS_DATA_PATH, "r") as read_file:
        json_data = json.load(read_file)

    to_df = []
    for line in json_data:
        for review in line["reviews"]:
            line_to_df = []
            line_to_df.append(line["name"])
            line_to_df.append(review["name"])
            line_to_df.append(review["rating"])
            line_to_df.append(review["title"])
            line_to_df.append(review["content"])
            to_df.append(line_to_df)
    reviews_df = pd.DataFrame(data=to_df, columns=columns)

    return reviews_df


@st.cache
def get_model():
    return torchmoji_emojis(PRETRAINED_PATH)


@st.cache
def get_vocab():
    with open(VOCAB_PATH, 'r') as f:
        vocabulary = json.load(f)
    return vocabulary


def top_elements(array, k):
    ind = np.argpartition(array, -k)[-k:]
    return ind[np.argsort(array[ind])][::-1]


def predict_emojis(texts, model, tokenizer):
    tokenized, _, _ = tokenizer.tokenize_sentences(texts)
    prob_list = model(tokenized)

    emoji_ids_list = [top_elements(prob, 10) for prob in prob_list]

    # map to emojis
    for i, emoji_ids in enumerate(emoji_ids_list):
        emojis = map(lambda x: EMOJIS[x], emoji_ids)
        st.write(emoji.emojize("{} {}".format(texts[i], ' '.join(emojis)), use_aliases=True))


def main():
    st.markdown("# 🎲 Game Reviews Sentiment Analysis With Emojis")
    game = st.selectbox("Choose game to analyse: ", GAME_LIST)

    reviews_df = load_data()

    maxlen = len(max(REVIEWS.values(), key=len))
    model = get_model()
    vocabulary = get_vocab()
    tokenizer = SentenceTokenizer(vocabulary, maxlen)
    predict_emojis([REVIEWS[game]], model, tokenizer)


if __name__ == "__main__":
    main()
