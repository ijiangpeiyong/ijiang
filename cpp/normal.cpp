// normal_distribution
#include <iostream>
#include <string>
#include <random>
#include<time.h>
#include<omp.h>
#include<openacc.h>

int main()
{
  const int nrolls=1e7;  // number of experiments
  const int nstars=100;    // maximum number of stars to distribute

  std::default_random_engine generator;
  std::normal_distribution<double> distribution(0.0,1.0);

  int p[10]={};

    clock_t t1=clock();
  
  for (int i=0; i<nrolls; ++i) {
    double number1 = distribution(generator);
    double number2 = distribution(generator);
    //std::cout<<number<<std::endl;
    //if ((number>=0.0)&&(number<10.0)) ++p[int(number)];
  }
    clock_t t2=clock();
  std::cout << "normal_distribution (5.0,2.0):" << std::endl;

  
  for (int i=0; i<10; ++i) {
    std::cout << i << "-" << (i+1) << ": ";
    std::cout << std::string(p[i]*nstars/nrolls,'*') << std::endl;
  }
  
  std::cout<<"S:"<<(double)(t2-t1)/CLOCKS_PER_SEC<<std::endl;

  return 0;
}
