@jit(nopython=True)
@jit(nogil=True)
@jit(cache=True)   To avoid compilation times each time you invoke a Python program, you can instruct Numba to write the result of function compilation into a file-based cache. This is done by passing cache=True:
@jit(nopython=True, parallel=True)  parallel=True and must be used in conjunction with nopython=True:

@generated_jit()  sometimes you want to write a function that has different implementations depending on its input types. 

ufunc: ufunc是universal function的缩写，意思是这些函数能够作用于narray对象的每一个元素上，而不是针对narray对象操作，numpy提供了大量的ufunc的函数。这些函数在对narray进行运算的速度比使用循环或者列表推导式要快很多，但请注意，在对单个数值进行运算时，python提供的运算要比numpy效率高。


@vectorize([float64(float64, float64)]) if you pass a list of signatures to the vectorize() decorator, your function will be compiled into a Numpy ufunc. In the 

        The vectorize() decorator supports multiple ufunc targets:

        Target	Description
        cpu	Single-threaded CPU
        parallel	Multi-core CPU
        cuda	    CUDA GPU

            from numba import vectorize, float64
            @vectorize([float64(float64, float64)])
            def f(x, y):
                return x + y

@guvectorize([(int64[:], int64, int64[:])], '(n),()->(n)')  the guvectorize() decorator takes the concept one step further and allows you to write ufuncs that will work on an arbitrary number of elements of input arrays, and take and return arrays of differing dimensions. The typical example is a running median or a convolution filter.
    可以用来迭代
        例子：
            guvectorize([(int64[:], int64, int64[:])], '(n),()->(n)')
            def g(x, y, res):
                for i in range(x.shape[0]):
                    res[i] = x[i] + y

            >>> a = np.arange(5)
            >>> a
            array([0, 1, 2, 3, 4])
            >>> g(a, 2)
            array([2, 3, 4, 5, 6])


@vectorize　：　Dynamic universal functions

@jitclass：用于类。。。

numba.cfunc() decorator creates a compiled function callable from foreign C code, using the signature of your choice.
        １．
        @cfunc("float64(float64, float64)")
        def add(x, y):
            return x + y

        print(add.ctypes(4.0, 5.0))  # prints "9.0"

        ２．．。



AOT：　没啥用　AOT compilation produces a compiled extension module which does not depend on Numba: you can distribute the module on machines which do not have Numba installed (but Numpy is required).

@stencil　重点关注！！！　　２Ｄ情况下，这是ｌａｐｌａｎｃｅ方程　　　
        from numba import stencil

        @stencil
        def kernel1(a):
            return 0.25 * (a[0, 1] + a[1, 0] + a[0, -1] + a[-1, 0])

numba.objmode() Warning : This feature can be easily mis-used. Users should first consider alternative approaches to achieve their intended goal before using this feature.


@njit 是 @jit(nopython=True) 的链接化　　can be used (the first is an alias of the second for convenience).

@njit(fastmath=True)：　True　是False的两倍快
    # without fastmath, this loop must accumulate in strict order
    # with fastmath, the reduction can be vectorized as floating point

parallel=True　　时候　ｎｏｇｉｌ


！Revisiting the reduce over sum example, assuming it is safe for the sum to be accumulated out of order, the loop in n can be parallelised through the use of prange. Further, the fastmath=True keyword argument can be added without concern in this case as the assumption that out of order execution is valid has already been made through the use of parallel=True (as each thread computes a partial sum).

###
parallel=True, fastmath=True
+ Intel SVML :::  conda install -c numba icc_rt 注意
#


CPU:
    The use of the parallel=True kwarg in @jit and @njit.
    The use of the target='parallel' kwarg in @vectorize and @guvectorize.

























