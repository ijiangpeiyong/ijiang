from multiprocessing import Process, Lock
from multiprocessing.sharedctypes import Value, Array
from ctypes import Structure, c_double


def modify(n):
    n.value **= 2

if __name__ == '__main__':
    lock = Lock()

    n = Value('i', 7)
    p = Process(target=modify, args=(n, ))
    p.start()
    p.join()

    print(n.value)




