import numpy as np

# This project requires a word embeddings file (e.g., GloVe vectors).
# You can download it from: https://nlp.stanford.edu/projects/glove/
# Place the file in this folder and rename it as: word_embeddings.txt


vocabulary_file = 'word_embeddings.txt'

# Read words
print('Read words...')
with open(vocabulary_file, 'r') as f:
    words = [x.rstrip().split(' ')[0] for x in f.readlines()]

# Read word vectors
print('Read word vectors...')
with open(vocabulary_file, 'r') as f:
    vectors = {}
    for line in f:
        vals = line.rstrip().split(' ')
        vectors[vals[0]] = [float(x) for x in vals[1:]]

vocab_size = len(words)
vocab = {w: idx for idx, w in enumerate(words)}
ivocab = {idx: w for idx, w in enumerate(words)}

# Vocabulary and inverse vocabulary (dict objects)
print('Vocabulary size')
print(len(vocab))

print(vocab['man'])
print(len(ivocab))
print(ivocab[10])

# W contains vectors for
print('Vocabulary word vectors')
vector_dim = len(vectors[ivocab[0]])
W = np.zeros((vocab_size, vector_dim))
for word, v in vectors.items():
    if word == '<unk>':
        continue
    W[vocab[word], :] = v
print(W.shape)


def find_analogy(word_x, word_y, word_z):
    vector_x = vectors[word_x]
    vector_y = vectors[word_y]
    vector_z = vectors[word_z]

    vector_x = np.array(vector_x)
    vector_y = np.array(vector_y)
    vector_z = np.array(vector_z)

    vect_X = vector_z + (vector_y - vector_x)

    words_to_consider = list(vocab.keys())
    word_to_remove1 = word_x
    word_to_remove2 = word_y
    word_to_remove3 = word_z

    words_to_consider.remove(word_to_remove1)
    words_to_consider.remove(word_to_remove2)
    words_to_consider.remove(word_to_remove3)


    distances = {w: np.linalg.norm(vect_X - W[vocab[w]]) for w in words_to_consider}
    sorted_distances = sorted(distances.items(), key=lambda x: x[1])
    print('word: ',sorted_distances[0][0], '    distance: ',sorted_distances[0][1])
    print('word: ',sorted_distances[1][0], '    distance: ',sorted_distances[1][1])

find_analogy('love', 'kiss', 'hate')
