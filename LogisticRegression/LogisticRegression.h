/*
*
*          logistic regression
* 
*
*/
class LogisticRegression 
{

public: 
  LogisticRegression(int, int, int);
  ~LogisticRegression();
  void train(int*, int*, double);
  void softmax(double*);
  void predict(int*, double*);
private:  
  int N;       // 样本数目
  int n_in;    // 输入层单元个数
  int n_out;   // 输出层单元个数
  double **W;  // 权重矩阵
  double *b;   // 偏置

};