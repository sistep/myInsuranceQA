import os
import sys
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing import text
from keras.utils import to_categorical
from keras.layers import Dense, Input, GlobalMaxPooling1D
from keras.layers import Conv1D, MaxPooling1D, Embedding
from keras.models import Model
import globalVals as gl

MAX_SEQUENCE_LENGTH = 200
# MAX_NB_WORDS = 20000
EMBEDDING_DIM = 200
POOL_SIZE=100

rawfile_train_path=gl.INSURANCE_DATA_RAW+gl.TRAIN_FILE(POOL_SIZE)
rawfile_valid_path=gl.INSURANCE_DATA_RAW+gl.VALID_FILE(POOL_SIZE)
rawfile_test_path=gl.INSURANCE_DATA_RAW+gl.TEST_FILE(POOL_SIZE)

rawfile_answers_path=gl.INSURANCE_DATA_RAW+gl.ANSWERS
rawfile_questions_path=gl.INSURANCE_DATA_RAW+gl.QUESTIONS

tokenfile_train_path=gl.INSURANCE_DATA_TOKEN+gl.TRAIN_FILE(POOL_SIZE)
tokenfile_valid_path=gl.INSURANCE_DATA_TOKEN+gl.VALID_FILE(POOL_SIZE)
tokenfile_test_path=gl.INSURANCE_DATA_TOKEN+gl.TEST_FILE(POOL_SIZE)

tokenfile_answers_path=gl.INSURANCE_DATA_TOKEN+gl.ANSWERS
tokenfile_questions_path=gl.INSURANCE_DATA_TOKEN+gl.QUESTIONS

ans_texts=[]
rawfile_answers=open(rawfile_answers_path)
print("loading answer raw file:",rawfile_answers.name)
for line in rawfile_answers:
    record = line.rstrip("\n").split('\t')
    ans_text =record[1]
    ans_texts.append(ans_text)
rawfile_answers.close()

ques_texts=[]
rawfile_questions=open(rawfile_questions_path)
print("loading question raw file:",rawfile_answers.name)
for line in rawfile_questions:
    record= line.rstrip("\n").split('\t')
    ques_text = record[1]
    ques_texts.append(ques_text)
rawfile_questions.close()

texts=[]
texts.extend(ans_texts)
texts.extend(ques_texts)
tokenizer=Tokenizer()
tokenizer.fit_on_texts(texts)
qus_seqs=tokenizer.texts_to_sequences(ques_texts)
ans_seqs=tokenizer.texts_to_sequences(ans_texts)
word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))

vocb_file=open(gl.INSURANCE_DATA_TOKEN+"vocabulary",'w')
print("writing vocabulary file......")
for word,i in word_index.items():
    vocb_file.write(str(i)+'\t'+word+'\n')
print("vocabulary finished")
vocb_file.close()

tokenfile_answer=open(tokenfile_answers_path,'w')
print("writing answer file......")
ans_index=0
for ans_sequence in ans_seqs:
    ans_index+=1
    line= str(ans_index)+'\t'
    for value in ans_sequence:
        line=line+str(value)+' '
    tokenfile_answer.write(line.rstrip(' ')+'\n')
tokenfile_answer.close()
print("answer file finished")

tokenfile_questions=open(tokenfile_questions_path,'w')
ques_index=0
for ques_sequence in qus_seqs:
    ques_index+=1
    line=str(ques_index)+'\t'
    for value in ques_sequence:
        line=line+str(value)+' '
    tokenfile_questions.write(line.rstrip(' ')+'\n')
tokenfile_questions.close()
print("question file finished")