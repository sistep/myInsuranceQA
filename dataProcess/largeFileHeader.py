import codecs

def show_file_header(file_path,rowcount):
    afile=codecs.open(file_path, 'r','utf-8')
    line=afile.readline().rstrip("\n")
    count=1
    while line is not None and line != "" and count<=rowcount:
        print(count,"-->",line)
        count += 1
        line=afile.readline().rstrip('\n')

if __name__ == "__main__":
    vector_400="../wiki.en.text.vector/wiki.en.vector.400"
    vector_200 = "../wiki.en.text.vector/wiki.en.text.vector"
    wiki_xml_file="../wiki.en.text.vector/enwiki-latest-pages-articles.xml"
    word2vec_npfile="../wiki.en.text.vector/wiki.en.model.wv.syn0.npy"
    rowcount=10
    show_file_header(vector_200,rowcount)