#include<openacc.h>
#include<iostream>
#include<time.h>
using namespace std;

int main()
{

    const int num=1e8;
    double *restrict data=new double[num];

    //restrict

    time_t t1,t2;

    t1=clock();
    #pragma acc kernels
    for(int i =0;i<num;i++){
        data[i]=0.1;

    }
    t2=clock();

    cout<<(double)(t2-t1)/CLOCKS_PER_SEC<<endl;

    delete [] data;
    cout<<"END"<<endl;
    return 0;
}




