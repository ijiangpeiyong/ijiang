#include <iostream>
#include <vector>
#include<time.h>

using namespace std;

int main()
{
    int M = 64;
    int N = 64;
    int L = 64;

    vector<vector<vector<double>>> A(M, vector<vector<double>>(N, vector<double>(L, 0)));

    for (int iM = 0; iM < M; iM++)
    {
        for (int iN = 0; iN < N; iN++)
        {
            for (int iL = 0; iL < L; iL++)
            {
                A[iM][iN][iL] = (double)(iM * iN * iL);
            }
        }
    }

    cout << A[0][0][0] << endl;


    time_t t1,t2;

    t1=closk();
    for (int iI = 1; iI < 1024; iI++)
    {
        for (int iM = 1; iM < M - 1; iM++)
        {
            for (int iN = 1; iN < N - 1; iN++)
            {
                for (int iL = 1; iL < L - 1; iL++)
                {
                    A[iM][iN][iL] = (A[iM + 1][iN][iL] + A[iM - 1][iN][iL] + A[iM][iN + 1][iL] + A[iM][iN - 1][iL] + A[iM][iN][iL + 1] + A[iM][iN][iL - 1]) / 6.;
                }
            }
        }
    }
    t2=closk();

    double dt=(double)(t2-t1)/CLOCKS_PER_SEC;

    cout<<'TIME: '<<dt<<endl;



    cout << "end" << endl;
    return 0;
}
