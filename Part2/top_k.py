import numpy as np
import math

words = np.loadtxt('vocab.txt', dtype=str)
words_vectors = np.loadtxt('wordVectors.txt')

word_to_vec = {word: word_vec for word, word_vec in zip(words, words_vectors)}


def dist(u, v):
    distance = (np.dot(u, v)) / (((np.dot(u, u))**0.5) * ((np.dot(v, v))**0.5))
    distance = 1 if distance > 1 else distance
    distance = -1 if distance < -1 else distance
    t = math.acos(distance)
    return abs(t)
    # return abs(math.acos(1 if distance > 1 else -1 if distance < -1 else distance))


def most_similar(word, k):
        word1_vec = word_to_vec[word]

        dist_to_word_data = [(dist(word1_vec, word2_vec), word2) for word2, word2_vec in word_to_vec.items()]
        dist_to_word_data.sort(key=lambda x: x[0])

        # first word is the given word itself so return elements 1 to k + 1
        return dist_to_word_data[1:k + 1]


def main():
    print(most_similar('dog', 5))
    print(most_similar('england', 5))
    print(most_similar('john', 5))
    print(most_similar('explode', 5))
    print(most_similar('office', 5))


if __name__ == '__main__':
    main()
