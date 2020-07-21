import string
import requests
import itertools
import random
import os
import re


def createRandomSubstitution() -> dict:
    alphabet1 = list(string.ascii_uppercase)
    alphabet2 = list(string.ascii_uppercase)
    random.shuffle(alphabet2)
    return dict(zip(alphabet1, alphabet2))


def reverseDict(dictionary1: dict) -> dict:
    keys = dictionary1.keys()
    values = dictionary1.values()
    return dict(zip(values, keys))


def obtainText(url: str, filename: str) -> str:
    if not os.path.exists(filename):
        text = fetchText(url)
        saveText(filename, text)

def fetchText(url: str) -> str:
    r = requests.get(url)
    return r.content.decode()


def saveText(filename: str, text: str):
    with open(filename, 'w') as f:
        f.write(text)


def string2map(encoddedAlphabet: str) -> dict:
    alphabet = list(string.ascii_uppercase)
    d = dict(zip(alphabet, encoddedAlphabet))
    return d


def createNGram(n: int) -> list:
    combination = [''.join(elem) for elem in itertools.product(string.ascii_uppercase, repeat=n)]
    return combination


def encodeMessage(text: str, cypher: dict) -> str:
    text = text.upper()
    regex = re.compile('[^a-zA-Z]')
    text = regex.sub(' ', text)

    codedMsg = []
    for char in text:
        codedChar = char
        if char in cypher:
            codedChar = cypher[char]
        codedMsg.append(codedChar)

    return ''.join(codedMsg)


def decodeMessage(text: str, cypher: dict) -> str:
    decodedMsg = []
    for char in text:
        decodedChar = char
        if char in cypher:
            decodedChar = cypher[char]
        decodedMsg.append(decodedChar)

    return ''.join(decodedMsg)