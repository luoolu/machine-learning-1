/*++++++++++++++++++++++++++++++++++++++++++++++++++++++*/
/*                   BP网络类                            */
/*                                                      */
/*                      writed by huqunwei，2015.3.5    */
/*++++++++++++++++++++++++++++++++++++++++*/
#ifndef _NEURALNET_H_
#define _NEURALNET_H_

#include <vector>
#include <cmath>
#include "Neuron.h"


using namespace std;

#define BIAS  1 
typedef vector<double> iovector;

/* 初始化权重0~0.05 */
inline double uniform(double _min, double _max)
{
	return rand()/(RAND_MAX + 1.0)*( _max - _min ) + _min;
}


class CNeuralNet 
{
private:
	int           m_nInput;        // 输入维度
	int           m_nOutput;       // 输出维度
	int           m_nNeuronsPerLyr;// 隐层神经元个数
	int           m_nHiddenLayer;  // 隐层层数


	double        m_dErrorSum;    // 误差

	NeuronLayer*  m_pHiddenLyr;   // 隐层
	NeuronLayer*  m_pOutLyr;      // 输出层

public:
	CNeuralNet(int nInput, int nOutput, int nNeuronsPerLyr, int nHiddenLayer = 1); // 构造函数
	~CNeuralNet();                                                                                              // 析构函数

	bool CalculateOutput(vector<double> Input, vector<double>& output);      //前向传播

	bool TrainEpoch(vector<iovector>& SetIn, vector<iovector>& SetOut, double LearnRate ); // 后向传播误差

	double GetErrorSum(){ return m_dErrorSum;}  // 误差

private:
	void CreateNetWork();

	void InitializeNetwork();

	double Sigmoid(double net)
	{
		return 1.0/(1 + exp(-net));
	}

};


#endif  // NeuralNet.h