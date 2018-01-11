import numpy as np

pred_file=open("simpleCNN/SCNN-mse-01-07_14-45.pred")
true_file=open("simpleCNN/SCNN-mse-01-07_14-45.true")

y_pred=np.loadtxt(pred_file)

y1=y_pred[0]
print("y1=",y1)
for label in y_pred:
    if label!=y1:
        print(label)

