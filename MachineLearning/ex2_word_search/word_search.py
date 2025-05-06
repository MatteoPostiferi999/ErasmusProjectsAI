import numpy as np

# This project requires a word embeddings file (e.g., GloVe vectors).
# You can download it from: https://nlp.stanford.edu/projects/glove/
# Place the file in this folder and rename it as: word_embeddings.txt


# type the word and press enter!

def find_nearest_neighbors(word, num_neighbors=3):
    word_index = vocab.get(word)
    input_vector = W[word_index]

    words_to_consider = list(vocab.keys())

    distances = {w: np.linalg.norm(input_vector - W[vocab[w]]) for w in words_to_consider}
    sorted_distances = sorted(distances.items(), key=lambda x: x[1])
    return [w for w, d in sorted_distances[:num_neighbors]]


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

while True:
    input_term = input("\nEnter word (EXIT to break): ")
    if input_term == 'EXIT':
        break
    else:
        nearest_neighbors = find_nearest_neighbors(input_term)

        print("\n                               Word       Distance\n")
        print("---------------------------------------------------------\n")
        for neighbor in nearest_neighbors:
            print(neighbor)
            distance = np.linalg.norm(W[vocab[input_term]] - W[vocab[neighbor]])
            print("%35s\t\t%f\n" % (neighbor, distance))

# type the word and press enter!
