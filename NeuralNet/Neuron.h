//////////////////////////////////////////////////////
// Neuron.h
#ifndef __SNEURON_H__
#define __SNEURON_H__
  
#define NEED_MOMENTUM //if you want to add momentum, remove the annotation
  
#define MOMENTUM 0.6 //momentum coefficient, works on when defined NEED_MOMENTUM£¨¶¯Á¿Ïî£©
  
typedef double WEIGHT_TYPE; // define datatype of the weight
  
  
struct SNeuron{//neuron cell
         /******Data*******/
         int m_nInput; //number of inputs
         WEIGHT_TYPE* m_pWeights;  //weights array of inputs
#ifdef NEED_MOMENTUM
         WEIGHT_TYPE* m_pPrevUpdate; //record last weights update when momentum is needed
#endif
         double m_dActivation; //output value, through Sigmoid function
         double m_dError; //error value of neuron
  
         /********Functions*************/
         void Init(int nInput){
                   m_nInput= nInput + 1; //add a side term,number of inputs is actual number of actual inputs plus 1
                   m_pWeights= new WEIGHT_TYPE[m_nInput];//allocate for weights array
#ifdef NEED_MOMENTUM
                   m_pPrevUpdate= new WEIGHT_TYPE[m_nInput];//allocate for the last weights array
#endif
                   m_dActivation= 0; //output value, through SIgmoid function
                   m_dError= 0;  //error value of neuron
         }
  
         ~SNeuron(){
                   //releasememory
                   delete []   m_pWeights;
#ifdef NEED_MOMENTUM
                   delete []   m_pPrevUpdate;
#endif
         }
};//SNeuron
  
  
struct SNeuronLayer{//neuron layer
         /************Data**************/
         int m_nNeuron; //Neuron number of this layer
         SNeuron*m_pNeurons; //Neurons array
  
         /*************Functions***************/
         SNeuronLayer(int nNeuron, int nInputsPerNeuron){
                   m_nNeuron= nNeuron;
                   m_pNeurons= new SNeuron[nNeuron];  //allocatememory for nNeuron neurons
  
                   for(int i=0; i<nNeuron; i++){
                            m_pNeurons[i].Init(nInputsPerNeuron);  //initialize neuron
                   }
         }
         ~SNeuronLayer(){
                   delete[]m_pNeurons;  //release neurons array
         }
};//SNeuronLayer
  
#endif//__SNEURON_H__