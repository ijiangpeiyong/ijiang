import multiprocessing as mp
import time

def clock(interval):
    while True:
        print("The time is %s" % time.ctime())
        time.sleep(interval)

if __name__=='__main__':
    p=mp.Process(target=clock,args=(15,))
    p.start()





