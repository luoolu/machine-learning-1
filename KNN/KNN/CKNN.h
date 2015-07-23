//----------------------------------------------------------------------
//                             KNN分类器
//                    封装CKNN类，用于分类测试
//
//----------------------------------------------------------------------
//                                                       writed by huqunwei，2015.3.2                                                
//----------------------------------------------------------------------

#ifndef _CKNN_H_
#define _CKNN_H_

#include<string>
#include<vector>
#include<map>
#include<iostream>
#include<cmath>
#include<algorithm>


#define ATTR_NUM    4         //特征维数 

using namespace std;

//结构体
struct sample 
{
	string classLabel;                  //类别
	double features[ATTR_NUM]; //特征向量
};

// sort排序   比较函数
struct CmpByValue
{
	bool operator()  (const pair<int ,double>& lhs, const pair<int ,double>& rhs)
	{
		return lhs.second < rhs.second;
	}
};

//设计CKNN类
class CKNN
{
public:
	CKNN(int k); // 构造函数

	~CKNN();     // 析构函数

	int Load(const char* SampleSet); 

	double  Distance(struct sample v1, struct sample v2); 

	void get_all_distance(struct sample TestSample );

    string  get_max_freq_label(); 

	void  get_accuracy();                

public:
	vector<struct sample > TrainSet;              //训练集合
	vector<struct sample > TestSet;               //测试集合
	int k;                                                         //K近邻参数
	map<int, double> map_index_dis;            
    map<string, int> map_label_freq;           
	vector<pair<int ,double>> vec_index_dis; 
};

#endif