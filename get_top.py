import pandas as pd
import numpy as np


# пример функции для добычи эмодзи по названию
def get_emojis_from_game_name(df, game_name="The Witcher 3: Wild Hunt", top_n=5):
    return " ".join(df.loc[game_name].sort_values(ascending=False)[:top_n].index)


# пример функции для добычи эмодзи по жанрам
def get_emojis_from_genre(df, emoji_cols, game_genre="Action", top_n=5,):
    return " ".join(df.loc[df[game_genre] == 1, emoji_cols].sum(axis=0).sort_values(ascending=False).index[:top_n])


# пример функции для добычи эмодзи по жанрам из сгруппированного датасета
def get_top_emojis_by_genres_from_gr(df, game_genre="Action", top_n=5):
    return " ".join(df.loc[game_genre].sort_values(ascending=False)[:top_n].index)


def get_top_by_column(df, col, value, top_n=10):
    assert col != 'name'
    emoji_cols = [col for col in df.columns if not col.isascii()]
    num = df[[col, 'name']].groupby(col).count().values
    dist = df[[col] + emoji_cols].groupby(col).sum() / num * 100
    return pd.DataFrame(dist.loc[value].sort_values(ascending=False).iloc[:top_n])
