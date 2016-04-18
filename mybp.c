#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

#define DATA  820  /* number of neuron 每层神经元的数目 */
#define IN 2  /* 每个样本有多少输入变量 */
#define OUT 1  /* 每个样本有多少个输出变量 */
#define NEURON 45  /* 神经元数量 */
#define TRAINC 20000  /* 训练次数上限 */
/* 学习率 */
#define A  0.2
#define B  0.4
#define a  0.2
#define b  0.3

#define TRAIN_FILE_INPUT "./in.txt"
#define TRAIN_FILE_OUTPUT "./out.txt"

double data_in[DATA][IN];  /* 存储DATA个样本，每个样本IN个输入 */
double data_out[DATA][OUT];  /* 存储DATA个样本，每个样本OUT个输出 */
double input_weight[NEURON][IN];  /* 输入对神经元的权重 */
double output_weight[OUT][NEURON];  /* 神经元对输出的权重 */
double input_delta[NEURON][IN];  /* 输入权重的修正量 */
double output_delta[OUT][NEURON];  /* 输出权重的修正量 */
double activate[NEURON];  /* 神经元激活函数对外的输出 */
double output_data[OUT];  /* BP神经网络的输出 */
double error;  /* 误差 */
double max_in[IN], min_in[IN], max_out[OUT], min_out[OUT];

/* 写训练数据 */
void write_test() {

	FILE *fp_input, *fp_output;
	double opeator1, opeator2;
	int i;
	srand((unsigned int)time(NULL));
	if ((fp_input = fopen(TRAIN_FILE_INPUT, "w")) == NULL) {
		fprintf(stderr, "can not open the in file\n");
		exit(EXIT_FAILURE);
	}
	if ((fp_output = fopen(TRAIN_FILE_OUTPUT, "w")) == NULL) {
		fprintf(stderr, "can not open the out file\n");
		exit(EXIT_FAILURE);
	}

	for (i = 0; i < DATA; ++i) {
		opeator1 = rand() % 1000 / 100.0;
		opeator2 = rand() % 1000 / 100.0;
		fprintf(fp_input, "%lf  %lf\n", opeator1, opeator2);
		fprintf(fp_output, "%lf \n", opeator1 + opeator2);
	}
	fclose(fp_input);
	fclose(fp_output);
}

/* 读训练数据 */
void read_data() {

	FILE *fp_tmp;
	int i, j;
	if ((fp_tmp = fopen(TRAIN_FILE_INPUT,"r")) == NULL) {
		fprintf(stderr, "can not open the in file\n");
		exit(EXIT_FAILURE);
	}
	for (i = 0; i < DATA; ++i) {
		for (j = 0; j < IN; ++j)
			fscanf(fp_tmp, "%lf", &data_in[i][j]);
	}
	fclose(fp_tmp);

	if ((fp_tmp = fopen(TRAIN_FILE_OUTPUT, "r")) == NULL) {
		fprintf(stderr, "can not open the out file\n");
		exit(EXIT_FAILURE);
	}
	for (i = 0; i < DATA; ++i) {
		for (j = 0; j < OUT; ++j)
			fscanf(fp_tmp,"%lf",&data_out[i][j]);
	}
	fclose(fp_tmp);
}

/* 初始化BP神经网络 */
void init_bpnetwork() {

	int i, j;

	for (i = 0; i < IN; ++i) {
		min_in[i] = max_in[i] = data_in[0][i];
		for (j = 0; j < DATA; ++j) {
			max_in[i]=max_in[i] > data_in[j][i] ? max_in[i] : data_in[j][i];
			min_in[i]=min_in[i] < data_in[j][i] ? min_in[i] : data_in[j][i];
		}
	}

	for (i = 0; i < OUT; ++i) {
		min_out[i] = max_out[i] = data_out[0][i];
		for (j = 0; j < DATA; ++j) {
			max_out[i] = max_out[i] > data_out[j][i] ? max_out[i] : data_out[j][i];
			min_out[i] = min_out[i] < data_out[j][i] ? min_out[i] : data_out[j][i];
		}
	}

	for (i = 0; i < IN; ++i) {
		for (j = 0; j < DATA; ++j)
			data_in[j][i] = (data_in[j][i] - min_in[i] + 1) / (max_in[i] - min_in[i] + 1);
	}

	for (i = 0; i < OUT; ++i) {
		for (j = 0; j < DATA; ++j)
			data_out[j][i] = (data_out[j][i] - min_out[i] + 1) / (max_out[i] - min_out[i] + 1);
	}

	for (i = 0; i < NEURON; ++i) {	
		for (j = 0; j < IN; ++j) {	
			input_weight[i][j] = rand() * 2.0 / RAND_MAX - 1;
			input_delta[i][j] = 0;
		}
	}

	for (i = 0; i < OUT; ++i) {
		for (j = 0; j < NEURON; ++j) {
			output_weight[i][j] = rand() * 2.0 / RAND_MAX - 1;
			output_delta[i][j] = 0;
		}
	}
}

/* 计算输出 */
void comput_output(int var) {

	int i,j;
	double sum;
	for (i = 0; i < NEURON; ++i) {
		sum = 0;
		for (j = 0; j < IN; ++j)
			sum += input_weight[i][j] * data_in[var][j];

		activate[i] = 1 / (1 + exp(-1 * sum));
	}

	for (i = 0; i < OUT; ++i) {
		sum = 0;
		for (j = 0; j < NEURON; ++j)
			sum += output_weight[i][j] * activate[j];

		output_data[i]=sum;
	}
}

/* 反馈学习 */
void back_update(int var) {

	int i, j;
	double t;
	for (i = 0; i < NEURON; ++i) {
		t = 0;
		for (j = 0; j < OUT; ++j) {
			t += (output_data[j] - data_out[var][j]) * output_weight[j][i];

			output_delta[j][i] = A * output_delta[j][i] + B * (output_data[j]-data_out[var][j]) * activate[i];
			output_weight[j][i] -= output_delta[j][i];
		}

		for (j = 0; j < IN; ++j) {
			input_delta[i][j] = a * input_delta[i][j] + b * t * activate[i] * (1-activate[i]) * data_in[var][j];
			input_weight[i][j] -= input_delta[i][j];
		}
	}
}

double result(double var1,double var2) {
	int i,j;
	double sum;

	var1 = (var1 - min_in[0] + 1) / (max_in[0] - min_in[0] + 1);
	var2 = (var2 - min_in[1] + 1) / (max_in[1] - min_in[1] + 1);

	for (i = 0; i < NEURON; ++i) {
		sum = 0;
		sum = input_weight[i][0] * var1 + input_weight[i][1] * var2;
		activate[i] = 1 / (1 + exp(-1 * sum));
	}
	sum = 0;
	for (j = 0; j < NEURON; ++j)
		sum += output_weight[0][j] * activate[j];

	return sum * (max_out[0] - min_out[0] + 1) + min_out[0] - 1;
}

/*void write_neuron() {
	FILE *fp1;
	int i,j;
	if((fp1=fopen("./neuron.txt","w"))==NULL)
	{
		printf("can not open the neuron file\n");
		exit(0);
	}
	for (i = 0; i < NEURON; ++i)	
		for (j = 0; j < IN; ++j){
			fprintf(fp1,"%lf ",w[i][j]);
		}
	fprintf(fp1,"\n\n\n\n");

	for (i = 0; i < NEURON; ++i)	
		for (j = 0; j < OUT; ++j){
			fprintf(fp1,"%lf ",v[j][i]);
		}

	fclose(fp1);
}*/

void  train_network() {

	int i, j, time = 0;
	do {
		error = 0.0;
		for (i = 0; i < DATA; ++i) {
			comput_output(i);
			for (j = 0; j < OUT; ++j)
				error += fabs((output_data[j] - data_out[i][j]) / data_out[i][j]);
			back_update(i);
		}
		printf("%d  %lf\n",time, error / DATA);
		++time;
	} while (time < TRAINC && error / DATA > 0.01);
}



int main(void)
{
	write_test();
	read_data();
	init_bpnetwork();
	train_network();
	printf("%lf \n",result(6,8) );
	printf("%lf \n",result(2.1,7) );
	printf("%lf \n",result(4.3,8) );
	//writeNEURON();
	return 0;
}
