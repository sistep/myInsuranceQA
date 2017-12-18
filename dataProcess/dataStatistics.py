import globalVals as gl

statics_file=open(gl.INSURANCE_DATA_RAW+"dataStatics.txt","a")

answerFile=open(gl.INSURANCE_DATA_RAW+gl.ANSWERS)
line=answerFile.readline().rstrip("\n")
maxLength=0
lineCount=0
totalLength=0
while line is not None and line != "":
    lineCount+=1
    lineLength=len(line.split("\t")[1].split(" "))
    totalLength+=lineLength
    if lineLength>=maxLength:
        maxLength=lineLength
    line=answerFile.readline()
avgLength=totalLength/lineCount
result="answers:"+str(lineCount)+"\ttotalLength:"+str(totalLength)+"\tmaxLength:"+str(maxLength)+"\tavgLength:"+str(avgLength)
statics_file.write(result+"\n")
print(result)

questionFile=open(gl.INSURANCE_DATA_RAW+gl.QUESTIONS,"r")
maxLength=0
lineCount=0
totalLength=0
line=questionFile.readline()
while line is not None and line != "":
    lineCount+=1
    lineLength=len(line.split("\t")[1].split(" "))
    totalLength+=lineLength
    if lineLength>=maxLength:
        maxLength=lineLength
    line=questionFile.readline()
avgLength=totalLength/lineCount
result="questions:"+str(lineCount)+"\ttotalLength:"+str(totalLength)+"\tmaxLength:"+str(maxLength)+"\tavgLength:"+str(avgLength)
statics_file.write(result+"\n")
print(result)

print("finished")