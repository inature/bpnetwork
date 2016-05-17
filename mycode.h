#ifndef MYCODE_H
#define MYCODE_H


#define DATA  27  /* 训练样本的数量 */
#define IN 4  /* 每个样本有多少输入变量 */
#define OUT 1  /* 每个样本有多少个输出变量 */
#define NEURON 38  /* 神经元数量 */

#define TRUE 1
#define FALSE 0

extern double data_out[DATA][OUT];  /* 存储DATA个样本，每个样本OUT个输出 */
extern double input_weight[NEURON][IN];  /* 输入对神经元的权重 */
extern double output_weight[OUT][NEURON];  /* 神经元对输出的权重 */
extern double output_data[OUT];  /* BP神经网络的输出 */

void comput_output(int var);

#endif /* MYCODE_H */
