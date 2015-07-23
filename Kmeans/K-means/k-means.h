/***************************************************************************
Module Name:
	KMeans

History:
	2003/10/16	Fei Wang
	2013 luxiaoxun
    modify by  huqunwei, 2015 
***************************************************************************/

//#pragma once
#ifndef K_MEANS_H
#define K_MEANS_H
#include <fstream>

class KMeans
{
public:
	enum InitMode
	{
		InitRandom,
		InitManual,
		InitUniform,
	}; //聚类中心初始化模式

	KMeans(int dimNum = 1, int clusterNum = 1);  // constructor
	~KMeans();  // destructor

	void SetMean(int i, const double* u){ memcpy(m_means[i], u, sizeof(double) * m_dimNum); } // 设置聚类中心
	void SetInitMode(int i)				{ m_initMode = i; }   // 设置初始化模式
	void SetMaxIterNum(int i)			{ m_maxIterNum = i; } // 迭代次数
	void SetEndError(double f)			{ m_endError = f; }   

	double* GetMean(int i)	{ return m_means[i]; }  
	int GetInitMode()		{ return m_initMode; }
	int GetMaxIterNum()		{ return m_maxIterNum; }
	double GetEndError()	{ return m_endError; }


	/*	SampleFile: <size><dim><data>...
		LabelFile:	<size><label>...
	*/
	void Cluster(const char* sampleFileName, const char* labelFileName); // cluster
	void Init(std::ifstream& sampleFile);
	void Init(double *data, int N); 
	void Cluster(double *data, int N, int *Label);  // cluster
	friend std::ostream& operator<<(std::ostream& out, KMeans& kmeans);

private: 
	int       m_dimNum;           //dimension
	int       m_clusterNum;       //聚类中心数目
	double**  m_means;       //聚类中心
	 
	int       m_initMode;         // 聚类中心初始化模式
	int       m_maxIterNum;		// The stopping criterion regarding the number of iterations
	double    m_endError;		// The stopping criterion regarding the error

	double GetLabel(const double* x, int* label); // 获取最近距离的标签
	double CalcDistance(const double* x, const double* u, int dimNum); // 计算欧式距离
};


#endif //k-means.h
