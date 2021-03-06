import json
import os

import emoji
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from get_top import get_emojis_from_game_name, get_top_emojis_by_genres_from_gr, get_top_by_column

from torchmoji.sentence_tokenizer import SentenceTokenizer
from torchmoji.model_def import torchmoji_emojis

from constants_ import (
    EMOJIS, MAX_TEXT_LEN, DEFAULT_REVIEW,
    VOCAB_PATH, DATA_FOLDER, DESCRIPTION,
    FLOAT_FORMATTER, COLUMNS_RENAME_DICT,
)
from download_model import download_pretrained, download_vocab


@st.cache
def load_data():
    games_df = pd.read_csv(DATA_FOLDER + 'Groupped_by_Name_df.csv', index_col=0).drop(columns='sum')
    genres_df = pd.read_csv(DATA_FOLDER + 'Groupped_by_Genres_df.csv', index_col=0).drop(columns='sum')
    full_data = pd.read_csv(DATA_FOLDER + 'emoji_data_without_text.csv').rename(COLUMNS_RENAME_DICT, axis=1)
    return games_df, genres_df, full_data


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

    emoji_dist_df = pd.DataFrame(pd.concat((games_df.loc[i, full_emojis]
                                            for i in games), axis=1)).iloc[:plot_emojis_num] * 20
    st.write(emoji_dist_df.iloc[:displayed_emojis_num].T.style.format(FLOAT_FORMATTER))
    emoji_dist_df = emoji_dist_df.reset_index()
    emoji_dist_df = emoji_dist_df.rename(columns={"index": "emoji"})
    emoji_dist_df = pd.melt(emoji_dist_df, id_vars=["emoji"], var_name="game", value_name="percentage")

    # Distribution plot
    st.markdown("### Emojis distribution plot")
    fig = px.bar(emoji_dist_df, x="emoji", y="percentage",
                 color='game', barmode='group',
                 height=400)
    st.plotly_chart(fig)


# Genre Distribution
def create_genre_dist(genres_df):
    genres = genres_df.index.values
    st.markdown("## 5 most popular emotions in genre")
    genre = st.selectbox("Choose genre to analyse: ", genres)
    genre_emojis = get_top_emojis_by_genres_from_gr(genres_df, genre)
    genre_empji_dist = genres_df.loc[[genre], genre_emojis.split()] * 20
    st.write(genre_empji_dist.style.format(FLOAT_FORMATTER))


# Other col distributions
def create_col_dist(full_data):
    st.markdown("## Aggregate by interesting game *features*")
    agg_cols = ['Review rating', 'Developer', 'Price category', 'Release year']
    st.markdown("In this section you can aggregate reviews by following params:\n")
    st.markdown('\n'.join([f'\t- {col}' for col in agg_cols]))
    agg_col = st.selectbox("Choose game feature to analyse: ", agg_cols)
    uniq_vals = full_data[agg_col].unique()
    value = st.selectbox(f"Choose {agg_col} to analyse:", sorted(uniq_vals))
    col_dist = get_top_by_column(full_data, agg_col, value)
    st.write(col_dist.T.iloc[:, :5].style.format(FLOAT_FORMATTER))

    fig = px.area(x=col_dist.index.tolist(), y=col_dist[value].tolist(),
                  labels={"x": "emoji", "y": "percentage"})
    st.plotly_chart(fig)


def load_pretrained():
    model = get_model()
    vocabulary = get_vocab()
    tokenizer = SentenceTokenizer(vocabulary, MAX_TEXT_LEN)
    return model, tokenizer


def predict_emojis(model, tokenizer):
    st.markdown("## Analyse your review")
    texts = [st.text_input("Write your review", DEFAULT_REVIEW)]

    tokenized, _, _ = tokenizer.tokenize_sentences(texts)
    prob_list = model(tokenized)
    emoji_ids = top_elements(prob_list[0], 5)
    confidences = prob_list[0][emoji_ids]

    emojis = list(map(lambda x: emoji.emojize(EMOJIS[x], use_aliases=True), emoji_ids))
    predictions = pd.DataFrame([confidences], columns=emojis)
    predictions.index = ["confidence"]
    st.write(predictions.style.format(FLOAT_FORMATTER))


def set_header():
    st.markdown("# 🎲 Game Reviews Sentiment Analysis With Emojis")
    authors = ['Maxim Sinyaev', 'Vyacheslav Yastrebov', 'Savkin Egor']
    np.random.shuffle(authors)
    st.markdown('**Authors**: ' + ', '.join([f'*{a}*' for a in authors]))


def set_references():
    resources = [
        'UX/UI — [streamlit](https://www.streamlit.io/)',
        'Model — [TorchMoji](https://github.com/huggingface/torchMoji)'
    ]
    st.markdown('## Resources:')
    resources = '\n'.join(f'\t- {resource}' for resource in resources)
    st.markdown(resources)


def main(top_n=10):
    set_header()
    # Loading data
    games_df, genres_df, full_data = load_data()
    # Loading torchMoji model
    model, tokenizer = load_pretrained()
    # Project description
    st.markdown(DESCRIPTION)
    # Games distribution
    games_distribution(games_df, top_n=top_n)
    # Top emojis by genre
    create_genre_dist(genres_df)
    # User choose col aggregation
    create_col_dist(full_data)
    # Predict user's review sentiments
    predict_emojis(model, tokenizer)
    # References
    set_references()


if __name__ == "__main__":
    main()
