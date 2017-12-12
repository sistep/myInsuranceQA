import logging
import os
import sys
import multiprocessing

from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence

if __name__ == '__main__':
    # program = os.path.basename(sys.argv[0])
    program="trainWord2vec"
    logger = logging.getLogger(program)

    logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
    logging.root.setLevel(level=logging.INFO)
    logger.info("running %s" % ' '.join("trainWord2vec"))

    # check and process input arguments
    # if len(sys.argv) < 4:
    #     print(globals()['__doc__'] % locals())
    #     sys.exit(1)
    # inp, outp1, outp2 = sys.argv[1:4]
    inp="../wiki.en.text.vector/wiki.en.text"
    outp1="../wiki.en.text.vector/wiki.en.model.100"
    outp2 = "../wiki.en.text.vector/wiki.en.vector.100"

    model = Word2Vec(LineSentence(inp), size=100, window=5, min_count=5,
                     workers=multiprocessing.cpu_count())

    # trim unneeded model memory = use(much) less RAM
    # model.init_sims(replace=True)
    model.save(outp1)
    model.wv.save_word2vec_format(outp2, binary=False)