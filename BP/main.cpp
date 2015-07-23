#include<iostream>
#include<string>
#include"NeuralNet.h"
// 提取类别
void GetLabel(vector<double> output, string& label);
//加载训练与测试集合
int Load(string filename , vector<iovector>& TrainSetIn,vector<iovector>& TrainSetOut,
	          vector<iovector>& TestSetIn,vector<iovector>& TestSetOut );


int main()
{
	string filename= "iris.data.txt";
	vector<iovector> TrainSetIn; // 训练特征集合
	vector<iovector> TrainSetOut;// 训练标注集合
	vector<iovector>  TestSetIn;  // 测试集合
	vector<iovector>  TestSetOut;
	if(0 != (Load(filename,TrainSetIn,TrainSetOut,TestSetIn,TestSetOut)))
	{
		return 1;
	}
	/* 初始化网络 */
	CNeuralNet  mynet( 4, 3, 10, 1);// 输入层维度，输出层维度，隐层维度，隐层层数
	/* 训练网络 */
	for(int epoch = 0; epoch < 1000; ++epoch )
    mynet.TrainEpoch(TrainSetIn, TrainSetOut, 0.4);
	
	
	/* 测试 */
	double accuracy = 0;  
	 int positive = 0;
	for(int i = 0; i < TestSetIn.size(); ++i )
	{
	      vector<double> output;
	     if(mynet.CalculateOutput(TestSetIn[i],output))
	     {
		       string TestLabel;
			   string TrueLabel;

			   GetLabel(TestSetOut[i],TrueLabel);
		       GetLabel(output,TestLabel);
		       cout<<"输出类别： "<<TestLabel <<" ("<<TrueLabel<<")" << endl;
			   if(0 ==(strcmp(TestLabel.c_str(), TrueLabel.c_str())))
				   positive ++;
	     }
	}
	accuracy = 1.0*positive / TestSetIn.size();
	cout << "分类正确率：accuracy =  " << accuracy*100 << "%" <<endl;
	system("pause");
	return 0;
}





/*获取最优类别*/
void GetLabel(vector<double> output, string& label)
{
	 int flag; // 最优类别索引标志位
     double  max = 0 ;
	 for(int i = 0 ; i < output.size(); ++i)
	 {
		 if( output[i] > max)
		 {
			 max = output[i];
			 flag = i;
		 }
	 }
	 switch(flag)
	 {
	 case 0:
		 label = "Iris-setosa";
		 break;
	 case 1:
		 label = "Iris-versicolor";
		 break;
	 case 2:
		 label = "Iris-virginica";
		 break;
	 default:
		 break;
	 }
}
/* 选出100组数据作为训练数据 ，剩下的50组作为测试数据*/
int Load(string filename , vector<iovector>& TrainSetIn,vector<iovector>& TrainSetOut,
	          vector<iovector>& TestSetIn,vector<iovector>& TestSetOut )
{
	FILE* fp;
	if(NULL == (fp  = fopen(filename.c_str(),"r")))	
	{
		cout << "文件打开失败！"<< endl;
		return 1;
	}
	else
	{
		iovector tmp1;
		iovector tmp2;
		int LineNum = 1;
		char bufLine[2048];
		while(NULL != fgets(bufLine,2048,fp))
		{
			char f0[10] = {0};
			char f1[10] = {0};
			char f2[10] = {0};
			char f3[10] = {0};
			char f4[20] = {0};
			tmp1.clear();
			sscanf(bufLine,"%[^','],%[^','],%[^','],%[^','],%s",f0,f1,f2,f3,f4 );
			tmp1.push_back( atof(f0) );
			tmp1.push_back( atof(f1) );
			tmp1.push_back( atof(f2) );
			tmp1.push_back( atof(f3) );
			if(0 != (LineNum%3))
			    TrainSetIn.push_back(tmp1); 
			else
				TestSetIn.push_back(tmp1);


			tmp2.clear();
			if(0 ==(strcmp("Iris-setosa",f4)))
			{
				tmp2.push_back(1);
				tmp2.push_back(0);
				tmp2.push_back(0);
			}
			else if( 0 == (strcmp("Iris-versicolor",f4)))
			{
				tmp2.push_back(0);
				tmp2.push_back(1);
				tmp2.push_back(0);
			}
			else if(0 == (strcmp("Iris-virginica",f4)))
			{
				tmp2.push_back(0);
				tmp2.push_back(0);
				tmp2.push_back(1);
			}
			if(0 != (LineNum%3))
			    TrainSetOut.push_back(tmp2);
			else
				TestSetOut.push_back(tmp2);

			LineNum ++;
		}
	}
	fclose(fp);
	return 0;

}



