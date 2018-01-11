import numpy as np

pred_file="simpleCNN/SCNN-hinge-01-06_18-26.pred"
true_file="simpleCNN/SCNN-hinge-01-06_18-26.true"

y_pred=np.loadtxt(pred_file)
y_true=np.loadtxt(true_file)

j=0
ques_count=0
right_ques=0
tp=False
fp=False
y_count=0

for i in range(len(y_true)):
    if y_pred[i][0]<0.5:
        print("->",y_pred[i],y_true[i])
        y_count+=1
        if y_true[i]==0:
            tp=True
        else:
            fp=True
    j+=1
    if j>=500:
        j=0
        ques_count+=1
        if tp==True and fp==False:
            right_ques+=1

print(right_ques/ques_count,right_ques,ques_count,y_count)