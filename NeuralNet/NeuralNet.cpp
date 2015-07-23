// NeuralNet.cpp: implementation of theCNeuralNet class.
//
//////////////////////////////////////////////////////////////////////
#include "NeuralNet.h"
#include <assert.h>
  
CNeuralNet::CNeuralNet(int nInput, int nOutput, int nNeuronsPerLyr, int nHiddenLayer){
         assert(nInput>0 && nOutput>0 && nNeuronsPerLyr>0 &&nHiddenLayer>0 );
         m_nInput= nInput;
         m_nOutput= nOutput;
         m_nNeuronsPerLyr= nNeuronsPerLyr;
         if(nHiddenLayer!= 1)
                   m_nHiddenLayer= 1;
         else
                   m_nHiddenLayer= nHiddenLayer; //temporarily surpport only one hidden layer
  
         m_pHiddenLyr= NULL;
         m_pOutLyr= NULL;
  
         CreateNetwork();   //allocate for each layer
         InitializeNetwork();  //initialize the whole network
}
  
CNeuralNet::~CNeuralNet(){
         if(m_pHiddenLyr!= NULL)
                   delete m_pHiddenLyr;
         if(m_pOutLyr!= NULL)
                   delete m_pOutLyr;
}
  
void CNeuralNet::CreateNetwork(){
         m_pHiddenLyr= new SNeuronLayer(m_nNeuronsPerLyr, m_nInput);
         m_pOutLyr= new SNeuronLayer(m_nOutput, m_nNeuronsPerLyr);
}
  
void CNeuralNet::InitializeNetwork(){
         int i, j;  //variables for loop
  
         /*usepresent time as random seed, so every time runs this programm can producedifferent random sequence*/
         //srand((unsigned)time(NULL) );
  
         /*initializehidden layer's weights*/
         for(i=0;i<m_pHiddenLyr->m_nNeuron; i++){
                   for(j=0;j<m_pHiddenLyr->m_pNeurons[i].m_nInput; j++){
                            m_pHiddenLyr->m_pNeurons[i].m_pWeights[j]= RandomClamped();

                            #ifdef NEED_MOMENTUM
                            /*whenthe first epoch train started, there is no previous weights update*/
                            m_pHiddenLyr->m_pNeurons[i].m_pPrevUpdate[j]= 0;
                            #endif
                   }       
         }
         /*initializeoutput layer's weights*/
         for(i=0;i<m_pOutLyr->m_nNeuron; i++){
                   for(int j=0; j<m_pOutLyr->m_pNeurons[i].m_nInput; j++){
                            m_pOutLyr->m_pNeurons[i].m_pWeights[j]= RandomClamped();
                            #ifdef NEED_MOMENTUM
                            /*whenthe first epoch train started, there is no previous weights update*/
                            m_pOutLyr->m_pNeurons[i].m_pPrevUpdate[j]= 0;
                            #endif
                   }
         }
  
         m_dErrorSum= 9999.0;  //initialize a large trainingerror, it will be decreasing with training
}
  
bool CNeuralNet::CalculateOutput(vector<double> input,vector<double>& output){
         if(input.size()!= m_nInput){ //input feature vector's dimention not equals to input of network
                   return false;
         }
         int i, j;
         double nInputSum;  //sum term
  
         /*compute hidden layer output*/
         for(i=0;i<m_pHiddenLyr->m_nNeuron; i++){
                   nInputSum= 0;
                   for(j=0;j<m_pHiddenLyr->m_pNeurons[i].m_nInput-1; j++){
                            nInputSum+= m_pHiddenLyr->m_pNeurons[i].m_pWeights[j] * input[j];
                   }
                   /*plusbias term*/
                   nInputSum+= m_pHiddenLyr->m_pNeurons[i].m_pWeights[j] * BIAS;
                   /*compute sigmoid fuction's output*/
                   m_pHiddenLyr->m_pNeurons[i].m_dActivation= Sigmoid(nInputSum);
         }
  
         /*compute output layer's output*/
         for(i=0;i<m_pOutLyr->m_nNeuron; i++){
                   nInputSum= 0;
                   for(j=0;j<m_pOutLyr->m_pNeurons[i].m_nInput-1; j++){
                            nInputSum+= m_pOutLyr->m_pNeurons[i].m_pWeights[j]
                                     *m_pHiddenLyr->m_pNeurons[j].m_dActivation;
                   }
                   /*plus bias term*/
                   nInputSum+= m_pOutLyr->m_pNeurons[i].m_pWeights[j] * BIAS;
                   /*computesigmoid fuction's output*/
                   m_pOutLyr->m_pNeurons[i].m_dActivation= Sigmoid(nInputSum);
                   /*saveit to the output vector*/
                   output.push_back(m_pOutLyr->m_pNeurons[i].m_dActivation);
         }
         return true;
}
  
bool CNeuralNet::TrainingEpoch(vector<iovector>&SetIn, vector<iovector>& SetOut, double LearningRate){
         int i, j, k;
         double WeightUpdate;  //weight's update value
         double err;  //error term
  
         /*increment'sgradient decrease(update weights according to each training sample)*/
         m_dErrorSum= 0;  // sum of error term
         for(i=0;i<SetIn.size(); i++){
                   iovector vecOutputs;
                   /*forwardlyspread inputs through network*/
                   if(!CalculateOutput(SetIn[i],vecOutputs)){
                            return false;
                   }
  
                   /*updatethe output layer's weights*/
                   for(j=0;j<m_pOutLyr->m_nNeuron; j++){
  
                            /*compute error term*/
                            err= ((double)SetOut[i][j]-vecOutputs[j])*vecOutputs[j]*(1-vecOutputs[j]);
                            m_pOutLyr->m_pNeurons[j].m_dError= err;  //record this unit's error
  
                            /*update sum error*/
                            m_dErrorSum+= ((double)SetOut[i][j] - vecOutputs[j]) * ((double)SetOut[i][j] -vecOutputs[j]);
  
                            /*update each input's weight*/
                            for(k=0;k<m_pOutLyr->m_pNeurons[j].m_nInput-1; k++){
                                     WeightUpdate= err * LearningRate * m_pHiddenLyr->m_pNeurons[k].m_dActivation;
#ifdef NEED_MOMENTUM
                                     /*update weights with momentum*/
                                     m_pOutLyr->m_pNeurons[j].m_pWeights[k]+=
                                               WeightUpdate+ m_pOutLyr->m_pNeurons[j].m_pPrevUpdate[k] * MOMENTUM;
                                     m_pOutLyr->m_pNeurons[j].m_pPrevUpdate[k]= WeightUpdate;
#else
                                     /*updateunit weights*/
                                     m_pOutLyr->m_pNeurons[j].m_pWeights[k]+= WeightUpdate;
#endif
                            }
                            /*biasupdate volume*/
                            WeightUpdate= err * LearningRate * BIAS;
#ifdef NEED_MOMENTUM
                            /*updatebias with momentum*/
                            m_pOutLyr->m_pNeurons[j].m_pWeights[k]+=
                                     WeightUpdate+ m_pOutLyr->m_pNeurons[j].m_pPrevUpdate[k] * MOMENTUM;
                            m_pOutLyr->m_pNeurons[j].m_pPrevUpdate[k]= WeightUpdate;
#else
                            /*update bias*/
                            m_pOutLyr->m_pNeurons[j].m_pWeights[k]+= WeightUpdate;
#endif
                   }//for out layer
  
                   /*update the hidden layer's weights*/
                   for(j=0;j<m_pHiddenLyr->m_nNeuron; j++){
  
                            err= 0;
                            for(int k=0; k<m_pOutLyr->m_nNeuron; k++){
                                     err+= m_pOutLyr->m_pNeurons[k].m_dError *m_pOutLyr->m_pNeurons[k].m_pWeights[j];
                            }
                            err*= m_pHiddenLyr->m_pNeurons[j].m_dActivation * (1 -m_pHiddenLyr->m_pNeurons[j].m_dActivation);
                            m_pHiddenLyr->m_pNeurons[j].m_dError= err;  //record this unit's error
  
                            /*update each input's weight*/
                            for(k=0;k<m_pHiddenLyr->m_pNeurons[j].m_nInput-1; k++){
                                     WeightUpdate= err * LearningRate * SetIn[i][k];
#ifdef NEED_MOMENTUM
                                     /*update weights with momentum*/
                                     m_pHiddenLyr->m_pNeurons[j].m_pWeights[k]+=
                                               WeightUpdate+ m_pHiddenLyr->m_pNeurons[j].m_pPrevUpdate[k] * MOMENTUM;
                                     m_pHiddenLyr->m_pNeurons[j].m_pPrevUpdate[k]= WeightUpdate;
#else
                                     m_pHiddenLyr->m_pNeurons[j].m_pWeights[k]+= WeightUpdate;
#endif
                            }
                            /*biasupdate volume*/
                            WeightUpdate= err * LearningRate * BIAS;
#ifdef NEED_MOMENTUM
                            /*updatebias with momentum*/
                            m_pHiddenLyr->m_pNeurons[j].m_pWeights[k]+=
                                     WeightUpdate+ m_pHiddenLyr->m_pNeurons[j].m_pPrevUpdate[k] * MOMENTUM;
                            m_pHiddenLyr->m_pNeurons[j].m_pPrevUpdate[k]= WeightUpdate;
#else
                            /*updatebias*/
                            m_pHiddenLyr->m_pNeurons[j].m_pWeights[k]+= WeightUpdate;
#endif
                   }//forhidden layer
         }//forone epoch
         return true;
}