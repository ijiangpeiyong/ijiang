#include <cstdio>
#include <cmath>
#include <iostream>

int main()
{

    double tol = 1e-6;
    int iter_max = 1e4;

    const int n = 32;
    const int m = 32;

    int *A = new int[m, n];

    for (int i = 0; i < m; i++)
    {
        for (int j = 0; j < n; j++)
        {

            /*
            if (j < n / 3)
                A[i, j] = 0.;
            else if (j > n * 2 / 3)
                A[i, j] = 0.;
            else
                A[i, j] = 0.5;
            */

            /*
            else if (i < m / 3)
            {
                std::cout << m * 2 / 3 << " " << i << "  " << j << " " << std::endl;
                A[i, j] = 0.3;
            }
            else if (i > m * 2 / 3)

                A[i, j] = 0.4;
            */

           /*
            if (i < m / 3)
                A[i, j] = 0.;
            else if (i > m * 2 / 3)
                A[i, j] = 0.;
            else
                A[i, j] = 0.5;
            */
           /*
           if ((i>10) &&(i<20))
           A[i, j] = 0.5;
           else
           A[i, j] = 0.;
            */

           A[i, j] = i+j;
           std::cout << i << "  " << j << " " <<i+j<< std::endl;


        }
    }

    /*
#pragma acc data copy(A, Anew)
    while ( error > tol && iter < iter_max )
    {
  error = 0.f;
 
#pragma omp parallel for shared(m, n, Anew, A)
#pragma acc kernels
  for( int j = 1; j < n-1; j++)
  {
for( int i = 1; i < m-1; i++ )
{
    Anew[j][i] = 0.25f * ( A[j][i+1] + A[j][i-1]
 + A[j-1][i] + A[j+1][i]);
    error = fmaxf( error, fabsf(Anew[j][i]-A[j][i]));
}
  }
 
#pragma omp parallel for shared(m, n, Anew, A)
#pragma acc kernels
  for( int j = 1; j < n-1; j++)
  {
for( int i = 1; i < m-1; i++ )
{
    A[j][i] = Anew[j][i];    
}
  }
 
  if(iter % 100 == 0) printf("%5d, %0.6fn", iter, error);
 
  iter++;
    }
 
    double runtime = GetTimer();
 
    printf(" total: %f sn", runtime / 1000.f);
}

*/
    for (int i = 0; i < m; i++)
    {
        for (int j = 0; j < n; j++)
        {

            std::cout << A[i, j] << " ";
        }
        std::cout << std::endl;
    }

    delete[] A;

    std::cout << "END" << std::endl;

    return 0;
}
