from nltk.parse.stanford import StanfordParser

if __name__ == '__main__':
    stanford_parser_dir = 'D:/workspace/stanford-parser-full-2015-04-20/'
    eng_model_path = "edu/stanford/nlp/models/lexparser/englishPCFG.ser.gz"
    my_path_to_models_jar = stanford_parser_dir + "stanford-parser-3.5.2-models.jar"
    my_path_to_jar = stanford_parser_dir + "stanford-parser.jar"

    parser = StanfordParser(model_path=eng_model_path, path_to_models_jar=my_path_to_models_jar, path_to_jar=my_path_to_jar)

    s = parser.raw_parse(
        "Targeted fiscal year 2015 voluntary programs, with eligibility limited by both grade and Air Force specialty codes, will be offered to help properly shape the force.")
    for line in s:
        for t in line:
            print(t)
