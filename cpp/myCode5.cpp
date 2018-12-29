
//rm *o; pgc++ -fast -Minfo==accel -ta=nvidia myCode5.cpp -o o; ./o

#include <iostream>
#include <vector>
#include <time.h>
#include <openacc.h>
#include <fstream>

using namespace std;

int main()
{
    int M = 8;
    int N = 8;
    int L = 8;

    double A[M][N][L];

    for (int iM = 0; iM < M; iM++)
    {
        for (int iN = 0; iN < N; iN++)
        {
            for (int iL = 0; iL < L; iL++)
            {

                A[iM][iN][iL] = (double)(iM * iN * iL);

                
                if (iM == 0)
                    A[iM][iN][iL] = 0.;
                if (iN == 0)
                    A[iM][iN][iL] = 0.;
                if (iL == 0)
                    A[iM][iN][iL] = 0.;
                if (iM == M - 1)
                    A[iM][iN][iL] = 0.;
                if (iN == N - 1)
                    A[iM][iN][iL] = 0.;
                if (iL == L - 1)
                    A[iM][iN][iL] = 0.;
                
            }
        }
    }

    time_t t1, t2;

    t1 = clock();


//#pragma acc kernels
//#pragma acc loop independent
    for (int iI = 1; iI < 2048; iI++)
    {
        for (int iM = 1; iM < M - 1; iM++)
        {
            for (int iN = 1; iN < N - 1; iN++)
            {
                for (int iL = 1; iL < L - 1; iL++)
                {
                    A[iM][iN][iL] = (A[iM + 1][iN][iL] + A[iM - 1][ iN][iL] + A[iM][iN + 1][iL] + A[iM][iN - 1][ iL] + A[iM][iN][iL + 1] + A[iM][iN][iL - 1]) / 6.;
                }
            }
        }
    }
    

    t2 = clock();

    double dt = (double)(t2 - t1) / CLOCKS_PER_SEC;

    cout << "TIME: " << dt << endl;

    ofstream myFile("pic3d");

    for (int numX = 0; numX < M; numX++)
    {
        for (int numY = 0; numY < N; numY++)
        {
            for (int numZ = 0; numZ < L; numZ++)
            {
                myFile << A[numX][numY][numZ]<<" ";
            }
            myFile << endl;
        }
        myFile << endl;
    }

    cout << "end 5" << endl;
    return 0;
}
