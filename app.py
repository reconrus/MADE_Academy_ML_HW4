import json
import os

import emoji
import numpy as np
import pandas as pd
import streamlit as st

from get_top import get_emojis_from_game_name, get_top_emojis_by_genres_from_gr

from torchmoji.sentence_tokenizer import SentenceTokenizer
from torchmoji.model_def import torchmoji_emojis

from constants import (
    EMOJIS, MAX_TEXT_LEN, DEFAULT_REVIEW,
    REVIEWS_DATA_PATH, VOCAB_PATH,
)
from download_model import download_pretrained, download_vocab


@st.cache
def load_data():
    games_df = pd.read_csv('./Groupped_by_Name_df.csv', index_col=0)
    genres_df = pd.read_csv('./Groupped_by_Genres_df.csv')
    genres_df.columns = ['genre', 'emojis']
    # genres_df.index.name = 'genre'
    return games_df, genres_df


@st.cache
def get_model():
    pretrained_path = download_pretrained()
    return torchmoji_emojis(pretrained_path)


@st.cache
def get_vocab():
    vocabulary_path = download_vocab()
    with open(vocabulary_path, 'r') as f:
        print(vocabulary_path)
        vocabulary = json.load(f)
    return vocabulary


def top_elements(array, k):
    ind = np.argpartition(array, -k)[-k:]
    return ind[np.argsort(array[ind])][::-1]


def predict_emojis(text, model, tokenizer):
    texts = [text]
    tokenized, _, _ = tokenizer.tokenize_sentences(texts)
    prob_list = model(tokenized)

    emoji_ids_list = [top_elements(prob, 10) for prob in prob_list]

    # map to emojis
    for i, emoji_ids in enumerate(emoji_ids_list):
        emojis = map(lambda x: EMOJIS[x], emoji_ids)
        # st.write(emoji.emojize("{} {}".format(texts[i], ' '.join(emojis)), use_aliases=True))
        st.write(emoji.emojize(' '.join(emojis), use_aliases=True))


def main():
    st.markdown("# 🎲 Game Reviews Sentiment Analysis With Emojis")
    # game = st.selectbox("Choose game to analyse: ", GAME_LIST)

    games_df, genres_df = load_data()
    games = games_df.index.values
    genres = genres_df.genre.values

    model = get_model()
    vocabulary = get_vocab()
    tokenizer = SentenceTokenizer(vocabulary, MAX_TEXT_LEN)

    game = st.selectbox("Choose game to analyse: ", games,)
    game_emojis = get_emojis_from_game_name(games_df, game)
    st.write(game_emojis)

    st.markdown("## Most popular emotions in genre")
    genre = st.selectbox("Choose genre to analyse: ", genres)
    genre_emojis = get_top_emojis_by_genres_from_gr(genres_df, genre)
    st.write(genre_emojis)

    st.markdown("## Analyse your review")
    user_text = st.text_input("Write your review", DEFAULT_REVIEW)
    predict_emojis(user_text, model, tokenizer)


if __name__ == "__main__":
    main()
