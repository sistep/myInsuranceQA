import logging
import numpy as np


logging.basicConfig(level=logging.DEBUG,
                format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                datefmt='%a, %d %b %Y %H:%M:%S',
                filemode='w')

list1=[1,2,3]
list2=[3,4,5]
a=1;
b=2;

list3=np.concatenate((list1,list2))

print(list3)
logging.error("aaaaa")
logging.warning("aaa")
logging.info("aa%d aa%d ",a,b)