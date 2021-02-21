import os
from textwrap import dedent

DESCRIPTION = dedent("""\
    If we want to choose a game, we often look at *reviews*. Reviews mainly describe the *emotions* the player received
    from the game. And what better way to reflect these emotions than Emoji?
    Our service allows you to convert reviews into emojis and view statistics on games in the context of Emoji, obtained
    from reviews on games.
    """)


EMOJIS = ":joy: :unamused: :weary: :sob: :heart_eyes: \
:pensive: :ok_hand: :blush: :heart: :smirk: \
:grin: :notes: :flushed: :100: :sleeping: \
:relieved: :relaxed: :raised_hands: :two_hearts: :expressionless: \
:sweat_smile: :pray: :confused: :kissing_heart: :heartbeat: \
:neutral_face: :information_desk_person: :disappointed: :see_no_evil: :tired_face: \
:v: :sunglasses: :rage: :thumbsup: :cry: \
:sleepy: :yum: :triumph: :hand: :mask: \
:clap: :eyes: :gun: :persevere: :smiling_imp: \
:sweat: :broken_heart: :yellow_heart: :musical_note: :speak_no_evil: \
:wink: :skull: :confounded: :smile: :stuck_out_tongue_winking_eye: \
:angry: :no_good: :muscle: :facepunch: :purple_heart: \
:sparkling_heart: :blue_heart: :grimacing: :sparkles:".split(' ')


# Paths to torchMoji model data
MODEL_FOLDER = "model"
PRETRAINED_PATH = os.path.join(MODEL_FOLDER, "pytorch_model.bin")
VOCAB_PATH = os.path.join(MODEL_FOLDER, "vocabulary.json")
WEIGHTS_DOWNLOAD_LINK = 'https://www.dropbox.com/s/q8lax9ary32c7t9/pytorch_model.bin?dl=1#'
VOCABULARY_DOWNLOAD_LINK = 'https://raw.githubusercontent.com/MaximSinyaev/torchMoji/master/model/vocabulary.json'

# Datasets
DATA_FOLDER = 'data/'
COLUMNS_RENAME_DICT = {
    'review_rating': 'Review rating',
    'developer': 'Developer',
    'price_bins': 'Price category',
    'release_date_year': 'Release year'
}


# Auxiliary
MAX_TEXT_LEN = 1000
MB_FACTOR = float(1 << 20)
DEFAULT_REVIEW = "Great game!"
FLOAT_FORMATTER = '{:,.2f}%'.format
