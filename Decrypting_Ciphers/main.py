from .genetic import GeneticAlgorithm
from .utils import createRandomSubstitution, encodeMessage, decodeMessage,reverseDict

original_message = '''I then lounged down the street and found,
as I expected, that there was a mews in a lane which runs down
by one wall of the garden. I lent the ostlers a hand in rubbing
down their horses, and received in exchange twopence, a glass of
half-and-half, two fills of shag tobacco, and as much information
as I could desire about Miss Adler, to say nothing of half a dozen
other people in the neighbourhood in whom I was not in the least
interested, but whose biographies I was compelled to listen to.
'''


def start():
    cypher = createRandomSubstitution()
    encryptedText = encodeMessage(original_message, cypher)
    print(encryptedText)
    # decypher = reverseDict(cypher)
    # decryptedText = decodeMessage(encryptedText, decypher)
    # print(decryptedText)
    alg = GeneticAlgorithm('https://lazyprogrammer.me/course_files/moby_dick.txt', 
                           'moby_dick.txt',
                           26)


if __name__ == "__main__":
    start()