#include <iostream>

int main()
{
    const int n = 32;
    const int m = 32;

    int* restrict A=new double[m]
    

    int **A = new int[m, n];

    for (int i = 0; i < m; i++)
    {
        for (int j = 0; j < n; j++)
        {
            A[i,j]=i+j;
            std::cout<<A[i,j]<<" ";
        }
        std::cout<<std::endl;
    }


    for (int i = 0; i < m; i++)
    {
        for (int j = 0; j < n; j++)
        {
            std::cout<<A[i,j]<<" ";
        }
        std::cout<<std::endl;
    }

    delete []A;

    return 0;
}