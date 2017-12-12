import os

def restore_file(source_filepath,target_filepath):
    source_file=open(source_filepath,'r')
    target_file=open(target_filepath,'w')
    line=source_file.readline().rstrip("\n")
    while line is not None and line!="":
        record = line.split("\t")
        index = record[0]
        words = record[1].split(" ")
        for i in range(len(words)):
            key = words[i]
            word = insuranceVocb.get(key)
            if word is not None:
                words[i] = word
        token = " "
        newline = index + "\t" + token.join(words)
        if len(record)>2:
            for i in range (2,len(record)):
                newline+="\t"+record[i]
        target_file.write(newline+"\n")
        line = source_file.readline().rstrip("\n")
    source_file.close()
    target_file.close()
    print("restored:",source_file.name," to:",target_file.name)
    return

if __name__=='__main__':
    insuranceVocb={}
    vocbFilePath="../insuranceData/encoded/vocabulary"
    vocbFile=open(vocbFilePath)
    line=vocbFile.readline()
    while line!=None and line!="":
        record=line.split("\t")
        insuranceVocb[record[0]]=record[1].rstrip("\n")
        line=vocbFile.readline()

    print("vocb size-->",len(insuranceVocb))
    vocbFile.close()

    encoded_file_dir="../insuranceData/encoded"
    raw_file_dir="../insuranceData/raw"
    qafiles = os.listdir(encoded_file_dir)
    for filename in qafiles:
        source_file_path=encoded_file_dir+"/"+filename
        if "encoded" in filename:
            target_file_path = raw_file_dir+"/"+filename[0:filename.rfind(".encoded")]
            restore_file(source_filepath=source_file_path ,target_filepath=target_file_path)

