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
PRETRAINED_PATH = "./torchMoji/model/pytorch_model.bin"
VOCAB_PATH = "./torchMoji/model/vocabulary.json"

# Dataset paths
REVIEWS_DATA_PATH = "./data/games.json"

# Mock data constants
# TODO delete in the final version, only for dev branch
GAME_LIST = ["Fallout 4", "Far Cry 3", "Cyberpunk 2077"]
REVIEWS = {
    "Fallout 4": "very boring building's everything",
    "Far Cry 3": "Nice game, Did I ever tell you what the definition of insanity is?",
    "Cyberpunk 2077": "bugs, bugs, bugs",
}
