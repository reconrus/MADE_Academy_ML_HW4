import json
import os

import emoji
import numpy as np
import pandas as pd
import streamlit as st

from get_top import get_emojis_from_game_name, get_top_emojis_by_genres_from_gr, get_top_by_column

from torchmoji.sentence_tokenizer import SentenceTokenizer
from torchmoji.model_def import torchmoji_emojis

from constants import (
    EMOJIS, REVIEWS_DATA_PATH,
    PRETRAINED_PATH, VOCAB_PATH,
    GAME_LIST, REVIEWS,  # TODO delete in the final version, only for dev branch
)

# Consts
DATA_FOLDER = 'data/'
FLOAT_FORMATTER = '{:,.2f}%'.format
COLUMNS_RENAME_DICT = {
    'review_rating': 'Review rating',
    'developer': 'Developer',
    'price_bins': 'Price category',
    'release_date_year': 'Release year'
}


@st.cache
def load_data():
    games_df = pd.read_csv(DATA_FOLDER + 'Groupped_by_Name_df.csv', index_col=0).drop(columns='sum')
    genres_df = pd.read_csv(DATA_FOLDER + 'Groupped_by_Genres_df.csv', index_col=0).drop(columns='sum')
    full_data = pd.read_csv(DATA_FOLDER + 'emoji_data_without_text.csv').rename(COLUMNS_RENAME_DICT, axis=1)
    # genres_df.columns = ['genre', 'emojis']
    return games_df, genres_df, full_data


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


def line_split(s, n):
    l = s.split()
    for i in range(1, len(l) + 1):
        if sum(map(len, l[:i])) + i - 1 > n:
            return ' '.join(l[:i - 1]), ' '.join(l[i - 1:])
    return s, ''


def text_split(s, max_len):
    res = list()
    while s:
        temp, s = line_split(s, max_len)
        res.append(temp)
    return res


def games_distribution(games_df, top_n=10):
    games = games_df.index.values
    displayed_emojis_num = 7
    plot_emojis_num = 10
    # Emojis distribution by game
    st.markdown('## Game reviews examination')
    game = st.selectbox("Choose game to analyse: ", games, index=0)
    game_emojis = get_emojis_from_game_name(games_df, game, top_n=top_n)
    compare = st.checkbox('Compare with another game')
    if compare:
        game_to_compare = st.selectbox("Choose game to analyse: ", games, index=1)
        game_to_compare_emojis = get_emojis_from_game_name(games_df, game_to_compare, top_n=top_n)
        full_emojis = list(set(game_emojis.split()) | set(game_to_compare_emojis.split()))
        games = [game, game_to_compare]
        st.markdown(f"### Comparing \"{game}\" and \"{game_to_compare}\" Emojis distribution")
    else:
        full_emojis = list(game_emojis.split())
        games = [game]
        st.markdown(f"### Top-{displayed_emojis_num} \"{game}\" Emojis distribution")

    emoji_dist_df = pd.DataFrame(pd.concat((games_df.loc[i, full_emojis] for i in games), axis=1)).iloc[:plot_emojis_num] * 20
    st.write(emoji_dist_df.iloc[:displayed_emojis_num].T.style.format(FLOAT_FORMATTER))

    # Distribution plot
    st.markdown("### Emojis distribution plot")
    st.line_chart(data=emoji_dist_df)


def set_description():
    DESCRIPTION = '''
If we want to choose a game, we often look at *reviews*. Reviews mainly describe the *emotions* the player received
from the game. And what better way to reflect these emotions than Emoji?

Our service allows you to convert reviews into emojis and view statistics on games in the context of Emoji, obtained
from reviews on games.
'''
    st.markdown(DESCRIPTION)


# Genre Distribution
def create_genre_dist(genres_df):
    genres = genres_df.index.values
    st.markdown("## 5 most popular emotions in genre")
    genre = st.selectbox("Choose genre to analyse: ", genres)
    genre_emojis = get_top_emojis_by_genres_from_gr(genres_df, genre)
    genre_empji_dist = genres_df.loc[[genre], genre_emojis.split()].style.format(FLOAT_FORMATTER)
    st.write(genre_empji_dist)

# Other col distributions
def create_col_dist(full_data):
    st.markdown("## Aggregate by interesting game *features*")
    agg_cols = ['Review rating', 'Developer', 'Price category', 'Release year']
    st.markdown("In this section you can aggregate reviews by following params:\n")
    st.markdown('\n'.join([f'\t- {col}' for col in agg_cols]))
    agg_col = st.selectbox("Choose game feature to analyse: ", agg_cols)
    uniq_vals = full_data[agg_col].unique()
    value = st.selectbox(f"Choose {agg_col} to analyse:", uniq_vals)
    col_dist = get_top_by_column(full_data, agg_col, value)
    st.write(col_dist.T.iloc[:, :5].style.format(FLOAT_FORMATTER))
    st.area_chart(data=col_dist)


def main(top_n=10):
    st.markdown("# 🎲 Game Reviews Sentiment Analysis With Emojis")
    # Loading data
    games_df, genres_df, full_data = load_data()
    # Project description
    set_description()
    # Games distribution
    games_distribution(games_df, top_n=top_n)
    # Top emojis by genre
    create_genre_dist(genres_df)
    # User choose col aggregation
    create_col_dist(full_data)


if __name__ == "__main__":
    main()
