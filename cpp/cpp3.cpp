// rm *o; pgc++ -fast -Minfo==accel myCode.cpp -o o; ./o

#include<openacc.h>
#include<iostream>
#include<time.h>
#include<random>
using namespace std;


int main()
{

    const int num=1e8;
    double *restrict data=new double[num];

    
  std::default_random_engine generator;
  std::normal_distribution<double> distribution(0.0,1.0);



    //restrict

    time_t t1,t2;

    t1=clock();
    #pragma acc kernels
    for(int i =0;i<num;i++){
        //data[i]=(double)i;
        data[i]=distribution(generator);

        //data[i,0]=distribution(generator);
        //data[i,1]=distribution(generator);

    }
    t2=clock();

    cout<<(double)(t2-t1)/CLOCKS_PER_SEC<<endl;

    
    for(int i =0; i<10; i++)
    {
        cout<<data[i,0]<<" "<<data[i]<<endl;
    }


    /*
    for(int i =0; i<10; i++)
    {
        cout<<data[i,0]<<" "<<data[i,1]<<endl;
    }
    */
    

    delete [] data;
    cout<<"END"<<endl;
    return 0;
}




