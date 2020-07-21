import numpy as np
import re
import string
import random
from .utils import obtainText


class GeneticAlgorithm():
    def __init__(self, url: str, filename: str, n: int):
        self.text = obtainText(url, filename)
        self.pi = self.createInitialDist(n)
        self.matrix = self.createMatrix(n)
        self.regex = re.compile('[^a-zA-Z]')
        self.computeProbabilities(filename)

        self.dnaPool = []
        self.initRandomDNA()

    def start(self, epochs: int):
        # Basic outline of the algorithms
        # epochs = 1000
        bestDNA = None
        bestMap = None
        maxScore = float('-inf')

        for i in range(epochs):
            if i>0:
                self.evolveOffspring(3)
            
            # score for each dna:
            dna2Score = {}
            for dna in self.dnaPool:
                currentMap = {}
                fitness_new = f(new_DNA)

            if score > maxScore:
                bestDNA = DNA_new
                bestMap = fitness_new


    def computeProbabilities(self, filename: str):
        for line in open(filename):
            line = line.rstrip()

            if line:
                line = self.regex.sub(' ', line)
                tokens = line.upper().split()
                for token in tokens:
                    i = token[0]
                    self.updateLetter(i)

                    for j in token[1:]:
                        self.updateTransition(i, j)
                        i = j

        # Normalize probabilities
        self.pi /= self.pi.sum()
        self.matrix /= self.matrix.sum(axis=1, keepdims=True)
        # print(self.matrix)
       
    def createMatrix(self, n: int) -> np.ndarray:
        return np.ones((n, n))

    def createInitialDist(self, n: int) -> np.ndarray:
        return np.zeros(n)

    def updateLetter(self, a: str) -> np.ndarray:
        i = ord(a)-65
        self.pi[i] += 1
        # return pi

    def updateTransition(self, a: str, b: str) -> np.ndarray:
        i = ord(a)-65
        j = ord(b)-65
        self.matrix[i, j] += 1
        # return matrix

    def getWordProb(self, word: str) -> np.float64:
        i = ord(word[0])-65
        logp = np.log(self.pi[i])

        for letter in word[1:]:
            j = ord(letter)-65
            logp += np.log(self.matrix[i, j])
            i = j
        
        return logp

    def getSequenceProb(self, words: list) -> np.float64:
        logp = 0
        for word in words:
            logp += self.getWordProb(word)
        
        return np.float64

    def initRandomDNA(self):
        for _ in range(20):
            dna = list(string.ascii_uppercase)
            dna = random.shuffle(dna)
            self.dnaPool.append(dna)

    def evolveOffspring(self, nChildren: int):
        # Create n children per offspring
        offspring = []
        for dna in self.dnaPool:
            for _ in range(nChildren):
                newDna = dna.copy()

                j = np.random.randint(len(newDna))
                k = np.random.randint(len(newDna))

                auxc = newDna[j]
                newDna[j] = newDna[k]
                newDna[k] = auxc
                offspring.append(newDna)
        self.dnaPool.append(offspring)
