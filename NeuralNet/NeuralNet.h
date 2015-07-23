// NeuralNet.h: interface for theCNeuralNet class.
//
//////////////////////////////////////////////////////////////////////
  
#ifndef __NEURALNET_H__
#define __NEURALNET_H__
  
#include <vector>
#include <math.h>
#include "Neuron.h"
using namespace std;
  
typedef vector<double> iovector;
#define BIAS 1 //bias term's coefficient w0
  
/*************Random functions initializingweights*************/
#define WEIGHT_FACTOR 0.1 //used to confineinitial weights
  
/*Return a random float between 0 to 1*/
inline double RandFloat(){ return(rand())/(RAND_MAX+1.0); }
  
/*Return a random float between -1 to 1*/
inline double RandomClamped(){ return WEIGHT_FACTOR*(RandFloat() - RandFloat()); }
  
class CNeuralNet{
private:
         /*Initial parameters, can not be changed throghout the whole training.*/
         int m_nInput;  //number of inputs
         int m_nOutput; //number of outputs
         int m_nNeuronsPerLyr; //unit number of hidden layer
         int m_nHiddenLayer; //hidden layer, not including the output layer
  
         /***Dinamicparameters****/
         double m_dErrorSum;  //one epoch's sum-error
  
         SNeuronLayer*m_pHiddenLyr;  //hidden layer
         SNeuronLayer*m_pOutLyr;     //output layer
  
public:
         /*
         *Constructorand Destructor.
         */
         CNeuralNet(int nInput, int nOutput, int nNeuronsPerLyr, int nHiddenLayer);
         ~CNeuralNet();
         /*
         *Computeoutput of network, feedforward.
         */
         bool CalculateOutput(vector<double> input,vector<double>& output);
         /*
         *Trainingan Epoch, backward adjustment.
         */
         bool TrainingEpoch(vector<iovector>& SetIn,vector<iovector>& SetOut, double LearningRate);
         /*
         *Geterror-sum.
         */
         double GetErrorSum(){ return m_dErrorSum; }
         SNeuronLayer* GetHiddenLyr(){ return m_pHiddenLyr; }
         SNeuronLayer* GetOutLyr(){ return m_pOutLyr; }
  
private:
         /*
         *Biuldnetwork, allocate memory for each layer.
         */
         void CreateNetwork();
         /*
         *Initializenetwork.
         */
         void InitializeNetwork();
         /*
         *Sigmoidencourage fuction.
         */
         double Sigmoid(double netinput){
                   double response = 1.0;  //control steep degreeof sigmoid function
                   return(1 / ( 1 + exp(-netinput / response) ) );
         }       
};
  
#endif //__NEURALNET_H__