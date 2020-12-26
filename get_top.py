import pandas as pd

# пример функции для добычи эмодзи по названию
def get_emojis_from_game_name(df, game_name="The Witcher 3: Wild Hunt", top_n=5):
    return " ".join(df.loc[game_name].sort_values(ascending=False)[:top_n].index)


# пример функции для добычи эмодзи по жанрам
def get_emojis_from_genre(df, emoji_cols, game_genre="Action", top_n=5,):
    return " ".join(df.loc[df[game_genre] == 1, emoji_cols].sum(axis=0).sort_values(ascending=False).index[:top_n])


# пример функции для добычи эмодзи по жанрам из сгруппированного датасета
def get_top_emojis_by_genres_from_gr(df, game_genre="Action", top_n=5):
    return " ".join(df.loc[df["genre"] == game_genre, "emojis"].values[0].split(",")[:top_n])
