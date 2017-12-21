import globalVals as gl

pools=[100,500,1000,1500]

quesFile=open(gl.INSURANCE_DATA_RAW+gl.QUESTIONS,'r')
quesIndex=0
quesDict={}
for line in quesFile:
    quesIndex+=1
    record=line.rstrip("\n").split('\t')
    quesDict[record[1]]=quesIndex
print(len(quesDict))


for poolSize in pools:
    rf_train=open(gl.INSURANCE_DATA_RAW+gl.TRAIN_FILE(poolSize))
    rf_valid = open(gl.INSURANCE_DATA_RAW + gl.VALID_FILE(poolSize))
    rf_test = open(gl.INSURANCE_DATA_RAW + gl.TEST_FILE(poolSize))

    tf_train = open(gl.INSURANCE_DATA_TOKEN + gl.TRAIN_FILE(poolSize),'w')
    tf_valid = open(gl.INSURANCE_DATA_TOKEN + gl.VALID_FILE(poolSize),'w')
    tf_test = open(gl.INSURANCE_DATA_TOKEN + gl.TEST_FILE(poolSize),'w')

    for line in rf_train:
       record=line.rstrip('\n').split('\t')
       ques_text=record[1]
       ques_index=quesDict.get(ques_text)
       newline=str(ques_index)+'\t'+record[2]+'\t'+record[3]
       tf_train.write(newline+'\n')
    rf_train.close()
    tf_train.close()

    for line in rf_valid:
        record = line.rstrip('\n').split('\t')
        ques_text = record[1]
        ques_index = quesDict.get(ques_text)
        newline = str(ques_index) + '\t' + record[2] + '\t' + record[3]
        tf_valid.write(newline + '\n')
    rf_valid.close()
    tf_valid.close()

    for line in rf_test:
        record = line.rstrip('\n').split('\t')
        ques_text = record[1]
        ques_index = quesDict.get(ques_text)
        newline = str(ques_index) + '\t' + record[2] + '\t' + record[3]
        tf_test.write(newline + '\n')
    rf_test.close()
    tf_test.close()

print("finished")