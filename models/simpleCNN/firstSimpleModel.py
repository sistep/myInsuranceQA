from keras.preprocessing import text
import globalVals as gl
import numpy as np
from keras.layers import Embedding

SEQUENCE_LENGTH=200
EMBEDDING_DIM=100
POOL_SIZE=100;

embedding_index={}
ans_index={}
qus_index={}
qap_index={}
qam_index={}
# load embedding index
vocbFile_demo = gl.INSURANCE_DATA_DEMO + "vocb.txt"
vocbFile = open(vocbFile_demo, "r")
line = vocbFile.readline().rstrip()
print("loading embedding index from:", vocbFile.name)
print(line)
line = vocbFile.readline().rstrip("\n")
while line is not None and line != "":
    values = line.split(" ")
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embedding_index[word] = coefs
    line = vocbFile.readline().rstrip("\n")
print("embedding index loaded")
vocbFile.close()

# load ans_index
ans_file = open(gl.INSURANCE_DATA_DEMO + gl.ANSWERS)
print("loading answer index from:", ans_file.name)
for line in ans_file:
    record = line.rstrip("\n").split('\t')
    index = record[0]
    ans_text = text.text_to_word_sequence(record[1])
    ans_index[index] = ans_text
print("answer index loaded")
ans_file.close()

# load qapool
qapool_file = open(gl.INSURANCE_DATA_DEMO + gl.TRAIN_FILE(POOL_SIZE))
print("loading qapool from:", qapool_file.name)
q_count = 0;
for line in qapool_file:
    q_count += 1
    record = line.rstrip("\n").split("\t")
    q_text = text.text_to_word_sequence(record[1])
    aps = record[2]
    ams = record[3]
    qus_index[q_count] = q_text
    qap_index[q_count] = np.asarray(aps.split(), dtype=np.int32)
    qam_index[q_count] = np.asarray(ams.split(), dtype=np.int32)
print("qusetion index、qap pairs、qam pairs lodaded")

# preparing the embedding layer
embedding_layer=Embedding()