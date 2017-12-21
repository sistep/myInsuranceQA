import logging
import time
import os
import sys
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing import text
from keras.utils import to_categorical,plot_model
from keras.layers import Dense, Input, GlobalMaxPooling1D
from keras.layers import Conv1D, MaxPooling1D, Embedding
from keras.models import Model
import globalVals as gl
import h5py


ISOTIMEFORMAT='%m-%d_%H-%M'
logger=logging.getLogger()
log_file=gl.LOG_PATH+'SCNN-'+time.strftime(ISOTIMEFORMAT,time.localtime())+'.log'
log_fh=logging.FileHandler(log_file)
log_ch=logging.StreamHandler()
log_formatter=logging.Formatter('%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s')
log_fh.setFormatter(log_formatter)
log_ch.setFormatter(log_formatter)
logger.addHandler(log_fh)
logger.addHandler(log_ch)
logger.setLevel(logging.DEBUG)
logger.info("starting...")

MAX_SEQUENCE_LENGTH = 200
EMBEDDING_DIM = 200
POOL_SIZE=100

vocabulary_file=open(gl.INSURANCE_DATA_TOKEN+"vocabulary")

ans_file=open(gl.INSURANCE_DATA_TOKEN+gl.ANSWERS)
ques_file=open(gl.INSURANCE_DATA_TOKEN+gl.QUESTIONS)

train_file=open(gl.INSURANCE_DATA_TOKEN+gl.TRAIN_FILE(POOL_SIZE))
valid_file=open(gl.INSURANCE_DATA_TOKEN+gl.VALID_FILE(POOL_SIZE))
test_file=open(gl.INSURANCE_DATA_TOKEN+gl.TEST_FILE(POOL_SIZE))


# second, prepare text samples and their labels
logger.info('Processing text dataset')

# load vocb of texts
logger.info("loading word index from %s" %vocabulary_file.name)
vocb={}
for line in vocabulary_file:
    record=line.split()
    vocb[record[0]]=record[1]
logger.info("loaded %d vocbs" % len(vocb))

# load answer sequences
ans_seqs=[]
logger.info("loading answer seq from: %s" %ans_file.name)
for line in ans_file:
    record = line.rstrip("\n").split('\t')
    ans_seqs.append(record[1].split())
ans_seqs=pad_sequences(ans_seqs,MAX_SEQUENCE_LENGTH)
logger.info("indexed %d answers" %len(ans_seqs))
ans_file.close()

# load questiong seqs
ques_seqs=[]
logger.info("loading question seqs from %s" % ques_file.name)
for line in ques_file:
    record=line.rstrip("\n").split('\t')
    ques_seqs.append(record[1].split())
ques_seqs=pad_sequences(ques_seqs,MAX_SEQUENCE_LENGTH)
logger.info("indexed %d questions" % len(ques_seqs))
ques_file.close()


# load train data
x_train=[]
y_train=[]
logger.info("loading train data from %s" %train_file.name)
for line in train_file:
    record=line.rstrip("\n").split('\t')
    ques_index=int(record[0])
    ans_p=map(int,record[1].split())
    ans_m=map(int,record[2].split())
    for index in ans_p:
        x_train.append(np.concatenate((ques_seqs[ques_index-1],ans_seqs[index-1])))
        y_train.append(1)
    for index in ans_m:
        x_train.append(np.concatenate((ques_seqs[ques_index - 1], ans_seqs[index - 1])))
        y_train.append(0)
train_file.close()
x_train=np.asarray(x_train)
y_train=to_categorical(y_train,2)
logger.info("loaded %d %d train data" ,len(x_train),len(y_train))

# load validation data
x_val=[]
y_val=[]
logger.info("loading validation data from %s" %valid_file.name)
for line in valid_file:
    record=line.rstrip("\n").split('\t')
    ques_index = int(record[0])
    ans_p=map(int,record[1].split())
    ans_m=map(int,record[2].split())
    for index in ans_p:
        x_val.append(np.concatenate((ques_seqs[ques_index-1],ans_seqs[index-1])))
        y_val.append(1)
    for index in ans_m:
        x_val.append(np.concatenate((ques_seqs[ques_index - 1], ans_seqs[index - 1])))
        y_val.append(0)
valid_file.close()
logger.info("loaded %d %d validation data" , len(x_val),len(y_val))
y_val=to_categorical(y_val,2)
x_val=np.asarray(x_val)

# load test data
x_test=[]
y_test=[]
logger.info("loading test data from %s" %test_file.name)
for line in test_file:
    record=line.rstrip("\n").split('\t')
    ques_index = int(record[0])
    ans_p=map(int,record[1].split())
    ans_m=map(int,record[2].split())
    for index in ans_p:
        x_test.append(np.concatenate((ques_seqs[ques_index-1],ans_seqs[index-1])))
        y_test.append(1)
    for index in ans_m:
        x_test.append(np.concatenate((ques_seqs[ques_index - 1], ans_seqs[index - 1])))
        y_test.append(0)
test_file.close()
logger.info("loaded %d %d test data" ,len(x_test),len(y_test))
y_test=to_categorical(y_test,2)
x_test=np.asarray(x_val)
# build index mapping words in the embeddings set
# to their embedding vector
logger.info('Indexing word vectors.')
embeddings_index = {}
f = open(gl.VOCB(EMBEDDING_DIM),encoding='utf-8')
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()
logger.info('Found %s word vectors.' % len(embeddings_index))

logger.info('shape of tenser:')
logger.info(x_train.shape)
logger.info("shape of labels:")
logger.info(y_train.shape)

# shuffle data
# indices=np.arange(x_train.shape[0])
# np.random.shuffle(indices)
# x_train=x_train[indices]
# y_train=y_train[indices]
#
# indices=np.arange(x_val.shape[0])
# np.random.shuffle(indices)
# x_val=x_val[indices]
# y_val=y_val[indices]
#
# indices=np.arange(x_test.shape[0])
# np.random.shuffle(indices)
# x_test=x_test[indices]
# y_test=y_test[indices]

# prepare embedding matrix
print('Preparing embedding matrix.')
embedding_matrix = np.zeros((len(vocb)+1, EMBEDDING_DIM))
for word,i in vocb.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector


# load pre-trained word embeddings into an Embedding layer
# note that we set trainable = False so as to keep the embeddings fixed
embedding_layer = Embedding(len(vocb)+1,
                            EMBEDDING_DIM,
                            weights=[embedding_matrix],
                            input_length=MAX_SEQUENCE_LENGTH*2,
                            trainable=False)

print('Training model.')

# train a 1D convnet with global maxpooling
sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH*2,), dtype='int32')
embedded_sequences = embedding_layer(sequence_input)
x = Conv1D(128, 5, activation='relu')(embedded_sequences)
x = MaxPooling1D(5)(x)
x = Conv1D(128, 5, activation='relu')(x)
x = MaxPooling1D(5)(x)
x = Conv1D(128, 5, activation='relu')(x)
x = GlobalMaxPooling1D()(x)
x = Dense(128, activation='relu')(x)
preds = Dense(2, activation='sigmoid')(x)

model = Model(sequence_input, preds)
model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['acc'])

model.fit(x_train, y_train,
          batch_size=128,
          epochs=10,
          shuffle=True,
          validation_data=(x_val, y_val))

model.save("./simpleCNN/simpleCNN-1221.h5")
plot_model(model,to_file="./simpleCNN/SCNN.png")

model.evaluate(x_test,y_test)


