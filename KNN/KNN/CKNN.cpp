#include "CKNN.h"

CKNN::CKNN(int k )
{
	this->k = k;
}


CKNN::~CKNN()
{

}

// 读取 样本集合 
// 按模3不为零的100组数据作为训练样本，剩下的50组作为测试样本
int CKNN::Load(const char*  SampleSet)
{
	int nRet = 0; // 返回码
	FILE * fp;
	if(NULL == (fp = fopen(SampleSet, "r")))
	{
		cout << "文件打开失败！"<<endl;
		return 1;
	}
	else
	{
		struct sample tmp;
		char bufLine[2048];
		int LineNum = 1;
		while(NULL != fgets(bufLine , 2048, fp))
		{

			char f0[10] = {0};
			char f1[10] = {0};
			char f2[10] = {0};
			char f3[10] = {0};
			char f4[20] = {0};
			sscanf(bufLine,"%[^','],%[^','],%[^','],%[^','],%s",f0,f1,f2,f3,f4 );
			tmp.features[0] = atof(f0);
			tmp.features[1] = atof(f1);
			tmp.features[2] = atof(f2);
			tmp.features[3] = atof(f3);
			tmp.classLabel = f4;
			if (0 != (LineNum%3))
			{
				TrainSet.push_back(tmp);
			}
			else
			{
				TestSet.push_back(tmp);
			}
			LineNum ++;
		}
	}
	fclose(fp);

	return  nRet;
}

// 计算两个样本之间的欧式距离
double CKNN::Distance(struct sample v1,struct sample v2)
{
	double dist;
	double sum = 0;
	for( int i = 0; i < ATTR_NUM; ++i)
	{
		sum +=  (v1.features[i] - v2.features[i])*(v1.features[i] - v2.features[i]);
	}
	dist = sqrt(sum);
	return dist;
}

//计算测试样本与训练集的距离
void CKNN::get_all_distance(struct sample TestSample)
{
	map_index_dis.clear();
	double distance;
	for(int i = 0; i < TrainSet.size(); ++i)
	{
		distance = Distance(TrainSet[i],TestSample);
		map_index_dis[i] = distance;
	}
}

// 统计频数，返回最大类别
string CKNN::get_max_freq_label()
{
	//vector<pair<int ,double>> vec_index_dis( map_index_dis.begin(), map_index_dis.end());
	vec_index_dis.clear();
	map<int,double >::iterator it = map_index_dis.begin();
	for(; it != map_index_dis.end(); ++it)
		vec_index_dis.push_back(*it);

	sort(vec_index_dis.begin(), vec_index_dis.end(), CmpByValue());

	map_label_freq.clear();

	for(int i = 0; i < k ; ++i )
	{
		/*
		cout << " the index = " << vec_index_dis[i].first << "  the distance = "  
		<< vec_index_dis[i].second << "  the label = " <<TrainSet[vec_index_dis[i].first].classLabel <<
		"  the coordiant ( "<<TrainSet[vec_index_dis[i].first].features[0]<<", "<< TrainSet[vec_index_dis[i].first].features[1]<<", "<<
		TrainSet[vec_index_dis[i].first].features[2]<<", " << TrainSet[vec_index_dis[i].first].features[3]<<" )" 
		<< endl;   
		*/

		map_label_freq[TrainSet[vec_index_dis[i].first ].classLabel] ++;
	}
	string label;
	map<string, int >::iterator  map_it = map_label_freq.begin();
	int max_freq = 0;
	while(map_it != map_label_freq.end())
	{
		if(map_it -> second > max_freq)
		{
			max_freq = map_it -> second;
			label = map_it -> first;
		}
		map_it ++;
	}
	//cout <<" the TestSample belongs to the " << label << "label" <<endl;
	return label;
}

// 计算分类正确率，并将结果写入到文件
void CKNN::get_accuracy()
{
	int positive = 0;
	double accuracy;
	FILE *fp;
	if (NULL == (fp = fopen("result.txt","w+t")))
	{
		cout<< "写入文件失败"<<endl;
	}
	fprintf(fp,"************************************程序说明************************************\n");
	fprintf(fp,"*************************采用KNN算法对iris.data分类****************************\n");
	fprintf(fp,"****共有150组数据，选择模3不等于0的100组作为训练集，剩下的50组做测试数据****\n");
	fprintf(fp,"********************************************************************************\n");
	fprintf(fp,"************************************实验结果*************************************\n");
	for(int i = 0; i < TestSet.size(); ++i)
	{
		fprintf(fp,"*************************************第%d组数据**********************************\n",i+1);
		get_all_distance(TestSet[i]);
		string label = get_max_freq_label();

		fprintf(fp,"KNN分类结果:    %s  （正确类别:    %s）\n",label.c_str(), TestSet[i].classLabel.c_str()  );
		if (0 == (strcmp(TestSet[i].classLabel.c_str(), label.c_str())))
			positive++;
		fprintf(fp,"%d-最近邻类别: \n", k);
		for(int i = 0; i < k; ++i )
		{
			fprintf(fp,"Distance:   %f\t  ClassLabel:    %s\n",vec_index_dis[i].second,TrainSet[vec_index_dis[i].first].classLabel.c_str());
		}
		fprintf(fp,"\n");
	
	}
	accuracy = positive*1.0 / TestSet.size();
	fprintf(fp, "************************************结果分析****************************************\n");
	fprintf(fp,"分类正确率：   %f %\n",accuracy*100);
	fclose(fp);
	cout << "分类正确率：accuracy =  " << accuracy*100 << "%" <<endl;
}




