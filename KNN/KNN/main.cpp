#include"CKNN.h"

int main()
{
	char * SampleSet = "iris.data.txt";
	int  k = 3;
	CKNN cknn(k);
	cknn.Load(SampleSet);
	cknn.get_accuracy();
	system("pause");
	return 0 ;
}
