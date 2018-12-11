import numpy as np
barrier=np.array([[1,0],[1,2],[1,3],[1,4],[2,2]])
girl=np.array([5,5])

'''
flag=1
if girl in barrier:
    flag=0
    print(girl)
    print(barrier)

print(flag)
'''
'''
for iBarrier in barrier:

    print(iBarrier)
    print(type(iBarrier))

    print((iBarrier==girl).all())
'''

#print(girl in barrier)


print(list(girl))
print(list(barrier))
