
DATA_FILE="F:/Workspaces/PythonWorkspaces/insuranceQAData/"

INSURANCE_DATA_ENCODED=DATA_FILE+"insuranceData/encoded/"

INSURANCE_DATA_RAW=DATA_FILE+"insuranceData/raw/"

INSURANCE_DATA_TOKEN=DATA_FILE+"insuranceData/token/"

INSURANCE_DATA_DEMO=INSURANCE_DATA_RAW+"demo/"

WORD2VEC_FILE=DATA_FILE+"wiki.en.text.vector/"

ANSWERS="InsuranceQA.label2answer.raw"

QUESTIONS="InsuranceQA.question.anslabel.raw"

def TEST_FILE(poolSize):
    return "InsuranceQA.question.anslabel.raw."+poolSize+".pool.solr.test"

def TRAIN_FILE(poolSize):
    return "InsuranceQA.question.anslabel.raw."+poolSize+".pool.solr.train"

def VALID_FILE(poolSize):
    return "InsuranceQA.question.anslabel.raw."+poolSize+".pool.solr.valid"