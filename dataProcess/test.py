import os

print('你好')

qafiles=os.listdir("../insuranceData")
for filename in qafiles:
    print(filename)
    if "test" in filename or "train" in filename or "valid" in filename:
        index=filename.rfind(".encoded")
        if index>-1:
            newfilename=filename[0:index]
            print(newfilename)
