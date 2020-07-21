import numpy as np
import re
import string
import random
import matplotlib.pyplot as plt
from .utils import obtainText, decodeMessage, createRandomSubstitution, encodeMessage


class GeneticAlgorithm():
    def __init__(self, originalText: str, url: str, filename: str, n: int):
        self.text = obtainText(url, filename)
        self.pi = self.createInitialDist(n)
        self.matrix = self.createMatrix(n)
        self.regex = re.compile('[^a-zA-Z]')
        self.computeProbabilities(filename)

        self.originalText = originalText.upper()
        self.encodedText = self.encodeText(self.originalText)
        print(self.encodedText)
        # self.trueMapping = trueMapping

        self.dnaPool = []
        self.initRandomDNA()


    def encodeText(self, originalMessage: str) -> str:
        self.trueMapping = createRandomSubstitution()
        return encodeMessage(originalMessage, self.trueMapping)

    def start(self, epochs: int):
        # Basic outline of the algorithms
        # epochs = 1000
        bestDNA = None
        self.bestMap = None
        maxScore = float('-inf')
        letters = list(string.ascii_uppercase)
        self.scores = np.zeros(epochs)
        

        for i in range(epochs):
            # print(f"===================EPOCH {i}=======================")
            if i > 0:
                # print("*******EVOLVING OFFSPRING*******************")
                self.evolveOffspring(3)

            # score for each dna:
            dna2Score = {}
            for dna in self.dnaPool:
                currentMap = dict(zip(letters, dna))
                # print(currentMap)
                decodedMessage = decodeMessage(self.encodedText, currentMap)
                score = self.getSequenceProb(decodedMessage)
                dna2Score[''.join(dna)] = score
                # print("**************************")

            if score > maxScore:
                bestDNA = dna
                self.bestMap = currentMap
                maxScore = score

            # Average score for n-generation:
            self.scores[i] = np.mean(list(dna2Score.values()))

            # Keep 5 fittest individuals
            sorted_dna = sorted(dna2Score.items(), key=lambda x: x[1], reverse=True)
            self.dnaPool = [list(k) for k, v in sorted_dna[:5]]

            if i % 200 == 0:
                print(f"iter: {i}, score: {self.scores[i]}, current max: {maxScore}")

        self.plot()


    def decypherText(self):
        decodedMessage = decodeMessage(self.encodedText, self.bestMap)
        print(f"Log Likelyhoodof decoded message: {self.getSequenceProb(decodedMessage)}")
        print(f"Log Likelyhoodof original message: {self.getSequenceProb(self.originalText)}")

        for key, value in self.trueMapping.items():
            pred = self.bestMap[value]
            if key != pred:
                print(f"True mapping: {key}, predicted mapping: {pred}")
        
        print("================================================================")
        print("Decoded message:")
        print(decodedMessage)
        print("----------------------------------------------------------------")
        print("True message:")
        print(self.originalText)

    def plot(self):
        plt.plot(self.scores)
        plt.show()

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

    def getSequenceProb(self, text: list) -> np.float64:
        logp = 0
        words = text.split()
        for word in words:
            logp += self.getWordProb(word)

        return logp

    def initRandomDNA(self):
        for _ in range(20):
            dna = list(string.ascii_uppercase)
            random.shuffle(dna)
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
        self.dnaPool.extend(offspring)
