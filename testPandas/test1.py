import pandas as pd
import numpy as np

numTest=10

A=pd.DataFrame(columns=['stateNow','action','reward','stateNext'])

for i in range(numTest):
    b=np.random.random((4))
    c=pd.Series(b,index=['stateNow','action','reward','stateNext'])
    A=A.append(c,ignore_index=True)




print('-'*20)
print(A)
A=A.drop([8,9])

print('-'*20)
print(A)

#print('-'*20)
#print(A)
