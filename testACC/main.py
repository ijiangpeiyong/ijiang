import os
folder='/home/pyong/ijiang/testACC/'

file='testAcc.cpp'

myCompile='pgc++ -Minfo=accel '+folder+file+' -o '+folder+'myTest'

myRun='sh '+folder+'myTest'

print(myRun)

os.system(myCompile)
os.system(myRun)




