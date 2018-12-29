#include<iostream>
#include<vector>

using namespace std;


int main(){
    int M=64;
    int N=64;
    int L=64;


    // typedef vector<int> v1d;
    // typedef vector<v1d> v2d;
    // typedef vector<v2d> v3d;
    // v3d A(L, v2d(N, v1d(M, 0)));


    // cout<<A[0][0][0]<<endl;


    vector<vector<vector<double>>> A(M,vector<vector<double>> (N,vector<double> (L,0)));


    for(int iM=0;iM<M;iM++){
        for(int iN=0;iN<N;iN++){
            for(int iL=0;iL<L;iL++){
                //std::cout<<A[iM,iN,iL] <<endl;
                //cout<<A[iM,iN,iL]<<endl;
            }
            //cout<<endl;
        }
        //cout<<endl;
    }

    cout<<A[0][0][0]<<endl;



    cout<<"end"<<endl;
    return 0;
}




