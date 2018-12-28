#include<stdio.h>
#ifdef _OPENACC
#include<openacc.h>
#endif
int main()
{
    #ifdef OPENACC
        printf("Number of device :%d\n",acc_get_num_devices(acc_device_not_host));
    #else
        printf("OpenACC is not support.\n");
    #endif
    return 0;
}



