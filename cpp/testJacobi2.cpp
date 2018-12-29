#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <openacc.h>

#if defined(_WIN32) || defined(_WIN64)
    #include <C:\Program Files (x86)\Windows Kits\10\Include\10.0.10150.0\ucrt\sys\timeb.h>    
    #define gettime(a)  _ftime(a)                                                           
    #define usec(t1,t2) ((((t2).time - (t1).time) * 1000 + (t2).millitm - (t1).millitm))    
    typedef struct _timeb timestruct;
#else
    #include <sys/time.h>
    #define gettime(a)  gettimeofday(a, NULL)
    #define usec(t1,t2) (((t2).tv_sec - (t1).tv_sec) * 1000 + ((t2).tv_usec - (t1).tv_usec)/1000) 
    typedef struct timeval timestruct;
#endif

#define ROW 8191
#define COL 1023

inline float uval(float x, float y)
{
    return x * x + y * y;
}

int main()
{
    const float width = 2.0, height = 1.0, hx = height / ROW, hy = width / COL, hx2 = hx * hx, hy2 = hy * hy;
    const float fij = -4.0f;
    const float maxIter = 100, errtol = 0.0f;
    const int COLp1 = COL + 1;
    const float c1 = hx2 * hy2 * fij, c2 = 1.0f / (2.0 * (hx2 + hy2));
    float *restrict u0 = (float *)malloc(sizeof(float)*(ROW + 1) * COLp1);
    float *restrict u1 = (float *)malloc(sizeof(float)*(ROW + 1) * COLp1);

    // 初始化
    int ix, jy;
    for (ix = 0; ix <= ROW; ix++)
    {
        u0[ix * COLp1 + 0] = u1[ix * COLp1 + 0] = uval(ix * hx, 0.0f);
        u0[ix * COLp1 + COL] = u1[ix * COLp1 + COL] = uval(ix * hx, COL * hy);
    }
    for (jy = 0; jy <= COL; jy++)
    {
        u0[jy] = u1[jy] = uval(0.0f, jy * hy);
        u0[ROW * COLp1 + jy] = u1[ROW * COLp1 + jy] = uval(ROW * hx, jy * hy);
    }
    for (ix = 1; ix < ROW; ix++)
    {
        for (jy = 1; jy < COL; jy++)
            u0[ix * COLp1 + jy] = 0.0f;
    }

    // 计算
    timestruct t1, t2;
    float /*uerr, temp,*/ *tempp;
    acc_init(4);        // 初始化设备，以便准确计时，4 代表 nVidia 设备
    gettime(&t1);

    for (int iter = 1; iter < maxIter; iter++)
    {
#pragma acc kernels copy(u0[0:(ROW + 1) * COLp1], u1[0:(ROW + 1) * COLp1])
        //uerr = 0.0f;
#pragma acc loop independent
        for (ix = 1; ix < ROW; ix++)
        {
            for (jy = 1; jy < COL; jy++)
            {
                u1[ix*COLp1 + jy] = c2 *
                    (
                        hy2 * (u0[(ix - 1) * COLp1 + jy] + u0[(ix + 1) * COLp1 + jy]) +
                        hx2 * (u0[ix * COLp1 + (jy - 1)] + u0[ix * COLp1 + (jy + 1)]) +
                        c1
                        );
                //temp = fabs(u0[ix * COLp1 + jy] - u1[ix * COLp1 + jy]);
                //uerr = max(temp, uerr);
            }
        }
        //printf("\niter = %d, uerr = %e\n", iter, uerr);
        //if (uerr < errtol)
        //    break;
        tempp = u0, u0 = u1, u1 = tempp;
    }
    gettime(&t2);
    acc_shutdown(4);    // 关闭设备，以便准确计时
    printf("\nElapsed time: %13ld ms.\n", usec(t1, t2));
    free(u0);
    free(u1);
    //getchar();
    return 0;
}