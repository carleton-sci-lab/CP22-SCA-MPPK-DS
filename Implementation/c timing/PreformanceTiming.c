#define _GNU_SOURCE
#include <stdio.h>
#include <intrin.h>
#include <time.h>

#pragma intrinsic(__rdtsc)

int main(){
clock_t start,end;
start =clock();
for(int i=0;i<10000000;i++){}
end=clock();
double total =(double)(end-start);
printf("%d\n",total);

}


/*

int main()
{	
	unsigned __int64 i;
	unsigned int ui;
	ui=2;
    i = __rdtscp(&ui);

	printf("%d\n",i);
	printf("%d\n",ui);
    return i,ui;
} */