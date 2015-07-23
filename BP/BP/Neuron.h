/*+++++++++++++++++++++++++++++++++++++++++++++++++++*/
/*         定义神经元结构体以及层结构体               */        
/*                                                  */
/*                  writed by huqunwei，2015.3.5    */
/*+++++++++++++++++++++++++++++++++++++++++++++++++++*/
#ifndef _NEURON_H_
#define _NEURON_H_

/*神经元 结构体*/
struct Neuron
{
	int     m_nInput;         //输入维度
	double* m_pWeights;       //权重

	double  m_dActivation;    //输出值
	double  m_dDelta;         //敏感度

	void Init(int nInput)
	{
		m_nInput = nInput + 1;
		m_pWeights = new double[m_nInput];
		m_dActivation = 0;
		m_dDelta = 0;
	}

	~Neuron()
	{
		delete []  m_pWeights;
	}
};


struct NeuronLayer
{
	int     m_nNeuron;        //神经元个数
	Neuron* m_pNeurons;  // 神经元   

	NeuronLayer(int nNeuron, int nInputsPerNeuron)
	{
		m_nNeuron = nNeuron;
		m_pNeurons = new Neuron[m_nNeuron];

		for(int i = 0; i < nNeuron; ++i)
		{
			m_pNeurons[i].Init(nInputsPerNeuron);
		}

	}
	~NeuronLayer()
	{
		delete [] m_pNeurons;
	}
};


#endif // Neuron.h
