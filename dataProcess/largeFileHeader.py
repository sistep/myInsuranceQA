import codecs
import globalVals as gl

def show_file_header(file_path,rowcount):
    afile=codecs.open(file_path, 'r','utf-8')
    line=afile.readline().rstrip("\n")
    count=1
    while line is not None and line != "" and count<=rowcount:
        print(count,"-->",line)
        count += 1
        line=afile.readline().rstrip('\n')

def small_demo_file(source_path,target_path,linecount):
    source_file=open(source_path,'r')
    target_file=open(target_path,'w')
    line=source_file.readline()
    count=1;
    while line is not None and line !="" and count<=linecount:
        target_file.write(line)
        line=source_file.readline()
        count+=1
    print("from:",source_path,"\n","to:",target_path,"\n","count:",linecount)

def get_demo_files():
    label2answer_demo = gl.INSURANCE_DATA_DEMO + "InsuranceQA.label2answer.raw"
    label2answer = gl.INSURANCE_DATA_RAW + "InsuranceQA.label2answer.raw"

    questions = gl.INSURANCE_DATA_RAW + "InsuranceQA.question.anslabel.raw"
    questions_demo = gl.INSURANCE_DATA_DEMO + "InsuranceQA.question.anslabel.raw"

    test100 = gl.INSURANCE_DATA_RAW + "InsuranceQA.question.anslabel.raw.100.pool.solr.test"
    test100_demo = gl.INSURANCE_DATA_DEMO + "InsuranceQA.question.anslabel.raw.100.pool.solr.test"

    train100 = gl.INSURANCE_DATA_RAW + "InsuranceQA.question.anslabel.raw.100.pool.solr.train"
    train100_demo = gl.INSURANCE_DATA_DEMO + "InsuranceQA.question.anslabel.raw.100.pool.solr.train"

    valid100 = gl.INSURANCE_DATA_RAW + "InsuranceQA.question.anslabel.raw.100.pool.solr.valid"
    valid100_demo = gl.INSURANCE_DATA_DEMO + "InsuranceQA.question.anslabel.raw.100.pool.solr.valid"

    small_demo_file(label2answer, label2answer_demo, 1000)
    small_demo_file(questions, questions_demo, 1000)
    small_demo_file(test100, test100_demo, 1000)
    small_demo_file(train100, train100_demo, 1000)
    small_demo_file(valid100, valid100_demo, 1000)

if __name__ == "__main__":
    vocbFile=gl.VOCB(200)
    vocbFile_demo=gl.INSURANCE_DATA_DEMO+"vocb.txt"
    small_demo_file(vocbFile,vocbFile_demo,100)