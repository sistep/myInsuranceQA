import gensim

model = gensim.models.KeyedVectors.load_word2vec_format("../wiki.en.text.vector/wiki.en.text.vector", binary=False)

print(model.most_similar("queen"))
print(model.most_similar("man"))
print(model.similarity("woman", "man"))