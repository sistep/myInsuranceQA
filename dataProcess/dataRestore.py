
insuranceVocb={}
vocbFilePath="../insuranceData/vocabulary"
vocbFile=open(vocbFilePath)
line=vocbFile.readline()
while line!=None and line!="":
    record=line.split("\t")
    insuranceVocb[record[0]]=record[1].rstrip("\n")
    line=vocbFile.readline()

print("size-->",len(insuranceVocb))
vocbFile.close()

answerEncodeFilePath="../insuranceData/InsuranceQA.label2answer.raw.encoded"
answerEncodeFile=open(answerEncodeFilePath,'r')
answerFilePath="../insuranceData/InsuranceQA.label2answer.raw"
answerFile=open(answerFilePath,'w')

line=answerEncodeFile.readline()
while line!=None and line!="":
    record=line.split("\t")
    index=record[0]
    answerEncoded=record[1].split(" ")

    for i in range(len(answerEncoded)):
        key=answerEncoded[i]
        word=insuranceVocb.get(key)
        if word is not None:
           answerEncoded[i]=word
    token=" "
    newline = index + "\t"+token.join(answerEncoded)
    answerFile.write(newline)
    line=answerEncodeFile.readline()
answerEncodeFile.close()
answerFile.close()
print("ok")
