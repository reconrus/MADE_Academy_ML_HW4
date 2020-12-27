from __future__ import print_function
import os
from subprocess import call

from constants import (
    MB_FACTOR,
    MODEL_FOLDER, PRETRAINED_PATH, VOCAB_PATH,
    WEIGHTS_DOWNLOAD_LINK, VOCABULARY_DOWNLOAD_LINK,
)


def download(path, link, name):
    if os.path.exists(path):
        print('{} file already exists at {}.'.format(name, path))
    else:
        print('Downloading the {} file from {}'.format(name, link))

        # downloading using wget due to issues with urlretrieve and requests
        sys_call = 'wget {} -O {}'.format(link, os.path.abspath(path))
        print("Running system call: {}".format(sys_call))
        call(sys_call, shell=True)

        print('Downloaded {} to {}'.format(name, path))
    return path


def download_pretrained():
    if not os.path.exists(MODEL_FOLDER):
        os.makedirs(MODEL_FOLDER)

    download(PRETRAINED_PATH, WEIGHTS_DOWNLOAD_LINK, "weights")
    if os.path.getsize(PRETRAINED_PATH) / MB_FACTOR < 80:
        raise ValueError("Download finished, but the resulting file is too small! " +
                         "It\'s only {} bytes.".format(os.path.getsize(PRETRAINED_PATH)))
    return PRETRAINED_PATH


def download_vocab():
    if not os.path.exists(MODEL_FOLDER):
        os.makedirs(MODEL_FOLDER)

    return download(VOCAB_PATH, VOCABULARY_DOWNLOAD_LINK, "vocabulary")