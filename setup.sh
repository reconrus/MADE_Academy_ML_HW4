#!/bin/bash
python3 -m venv env
source env/bin/activate
pip3 install wheel
pip3 install -r requirements.txt

## torchMoji installation
git clone https://github.com/MaximSinyaev/torchMoji.git torchMoji
pip3 install torchMoji/

## Pretrained model weigths download
cd torchMoji
python ./scripts/download_weights.py
cd ..

# echo "export PRETRAINED_PATH=\"./torchMoji/model/pytorch_model.bin\""
# echo "export VOCAB_PATH=\"./torchMoji/model/vocabulary.json\""

## Dataset download
[ ! -d ~/.kaggle ] && mkdir ~/.kaggle/
wget -O ~/.kaggle/kaggle.json "https://drive.google.com/uc?export=download&id=1sbrOZ6-lhTyeoP6TdXdkL8wocPlwbXE9"
chmod 600 ~/.kaggle/kaggle.json

[ ! -d data/ ] && mkdir data
cd data
if [ ! -f games.json ]; then
    pip install kaggle
    kaggle datasets download -d beastovest/gog-games-with-reviews
    unzip gog-games-with-reviews.zip
fi
cd ..
