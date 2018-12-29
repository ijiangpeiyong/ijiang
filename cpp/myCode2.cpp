#include <time.h>
#include <iostream>
#include <algorithm>
#include<fstream>

using namespace std;

void Init(double *A, int numA)
{
    for (int i = 0; i < numA; i++)
        A[i] = 0.1;
}

void Init3D(double *A, int gridX, int gridY, int gridZ)
{

    for (int numX = 0; numX < gridX; numX++)
    {
        for (int numY = 0; numY < gridX; numY++)
        {
            for (int numZ = 0; numZ < gridX; numZ++)
            {
                int numA = numZ * gridY * gridX + numY * gridX + numX;
                if ((numX < gridX / 3) || (numX > gridX / 3 * 2) || (numX < gridY / 3) || (numY > gridY / 3 * 2) || (numZ < gridZ / 3) || (numZ > gridZ / 3 * 2))
                    A[numA] = 0.;
                else
                    A[numA] = 0.5 + (double)(numX * numY * numZ) * 0.001;
            }
        }
    }
}

void Cout3D(double *A, int gridX, int gridY, int gridZ)
{

    for (int numX = 0; numX < gridX; numX++)
    {
        for (int numY = 0; numY < gridX; numY++)
        {
            for (int numZ = 0; numZ < gridX; numZ++)
            {
                int numA = numZ * gridY * gridX + numY * gridX + numX;
                cout << A[numA] << " ";
            }
            cout << endl;
        }
        cout << endl;
    }
}

void Cout(double *A, int numA)
{
    for (int i = 0; i < numA; i++)
        cout << A[i] << endl;
}

void CoutMax(double *A, int numA)
{
    cout << *max_element(A, A + numA) << endl;
}

double Max(double *A, int numA)
{
    return *max_element(A, A + numA);
}

void Jacobi(double *A, int gridX, int gridY, int gridZ)
{

#pragma acc kernels
#pragma acc loop independent
    for (int iter = 0; iter < 1024; iter++)
    {
        for (int numX = 1; numX < gridX - 1; numX++)
        {
            for (int numY = 1; numY < gridX - 1; numY++)
            {
                for (int numZ = 1; numZ < gridX - 1; numZ++)
                {
                    int numA110 = numZ * gridY * gridX + numY * gridX + (numX - 1);
                    int numA112 = numZ * gridY * gridX + numY * gridX + (numX + 1);
                    int numA101 = numZ * gridY * gridX + (numY - 1) * gridX + numX;
                    int numA121 = numZ * gridY * gridX + (numY + 1) * gridX + numX;
                    int numA011 = (numZ - 1) * gridY * gridX + numY * gridX + numX;
                    int numA211 = (numZ + 1) * gridY * gridX + numY * gridX + numX;

                    int numA = numZ * gridY * gridX + numY * gridX + numX;
                    A[numA] = (A[numA110] + A[numA112] + A[numA101] + A[numA121] + A[numA011] + A[numA211]) / 6.;
                }
            }
        }

    
        for (int numX = 0; numX < gridX; numX++)
        {
            for (int numY = 0; numY < gridX; numY++)
            {
                int numALast = (gridZ - 1) * gridY * gridX + numY * gridX + numX;
                int numFirst = numY * gridX + numX;
                A[numALast] = A[numFirst];
            }
        }
    
    }
}

int main()
{

    int gridX = 64;
    int gridY = 64;
    int gridZ = 64;

    const int numA = gridX * gridY * gridZ;

    double *A = new double[numA];

    Init3D(A, gridX, gridY, gridZ);

    //Cout3D(A, gridX, gridY, gridZ);
    //CoutMax(A, numA);

    //cout<<Max(A,numA)<<endl;

    time_t t1, t2;
    t1 = clock();
    Jacobi(A, gridX, gridY, gridZ);
    t2 = clock();
    double dt = (double)(t2 - t1) / CLOCKS_PER_SEC;

    ofstream myFile("pic3d");
    
    for (int numX = 0; numX < gridX; numX++)
    {
        for (int numY = 0; numY < gridX; numY++)
        {
            for (int numZ = 0; numZ < gridX; numZ++)
            {
                int numA = numZ * gridY * gridX + numY * gridX + numX;
                myFile << A[numA] << endl;
            }
            //myFile << endl;
        }
        //myFile << endl;
    }


    cout << "time  :  " << dt << endl;

    //CoutMax(A, numA);

    //Cout3D(A, gridX, gridY, gridZ);

    cout << "END" << endl;
    return 0;
}
