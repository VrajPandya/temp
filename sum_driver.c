#include<stdio.h>
#include"sum.h"

int main(int argc, const char** argv){
	double a = 20.0;
	double b = 5.0;

	double c;

	c = sum(a,b);

	printf("%f", c);

	return 0;
}
