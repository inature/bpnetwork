#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

#define DATA  27  /* 训练样本的数量 */
#define IN 4  /* 每个样本有多少输入变量 */
#define OUT 1  /* 每个样本有多少个输出变量 */
#define NEURON 38  /* 神经元数量 */
#define TRAINC 2000000000  /* 训练次数上限 */

/* 学习率 */
#define LEARN  0.2

/* 误差 */
#define ERROR 0.012

/* 存放训练数据的文件 */
#define TRAIN_FILE_INPUT "./train_in.txt"
#define TRAIN_FILE_OUTPUT "./train_out.txt"

/* 存放训练后的权值 */
#define NEURON_WEIGHT "./neuron.txt"

double data_in[DATA][IN];  /* 存储DATA个样本，每个样本IN个输入 */
double data_out[DATA][OUT];  /* 存储DATA个样本，每个样本OUT个输出 */
double input_weight[NEURON][IN];  /* 输入对神经元的权重 */
double output_weight[OUT][NEURON];  /* 神经元对输出的权重 */
double input_delta[NEURON][IN];  /* 输入权重的修正量 */
double output_delta[OUT][NEURON];  /* 输出权重的修正量 */
double activate[NEURON];  /* 神经元激活函数对外的输出 */
double output_data[OUT];  /* BP神经网络的输出 */
double error;  /* 误差 */
double max_in[IN], min_in[IN], max_out[OUT], min_out[OUT];  /* 训练数据的最值，用于归一化 */

/* 读训练数据 */
void read_data() {

	FILE *fp_tmp;
	int i, j;
	if ((fp_tmp = fopen(TRAIN_FILE_INPUT,"r")) == NULL) {
		fprintf(stderr, "can not open the in file\n");
		exit(EXIT_FAILURE);
	}
	for (i = 0; i < DATA; i++) {
		for (j = 0; j < IN; j++)
			fscanf(fp_tmp, "%lf", &data_in[i][j]);
	}
	fclose(fp_tmp);

	if ((fp_tmp = fopen(TRAIN_FILE_OUTPUT, "r")) == NULL) {
		fprintf(stderr, "can not open the out file\n");
		exit(EXIT_FAILURE);
	}
	for (i = 0; i < DATA; i++) {
		for (j = 0; j < OUT; j++)
			fscanf(fp_tmp, "%lf", &data_out[i][j]);
	}
	fclose(fp_tmp);
}

/* 初始化BP神经网络 */
void init_bpnetwork() {

	int i, j;

	for (i = 0; i < IN; i++) {
		min_in[i] = max_in[i] = data_in[0][i];
		for (j = 0; j < DATA; j++) {
			max_in[i]=max_in[i] > data_in[j][i] ? max_in[i] : data_in[j][i];
			min_in[i]=min_in[i] < data_in[j][i] ? min_in[i] : data_in[j][i];
		}
	}

	for (i = 0; i < OUT; i++) {
		min_out[i] = max_out[i] = data_out[0][i];
		for (j = 0; j < DATA; j++) {
			max_out[i] = max_out[i] > data_out[j][i] ? max_out[i] : data_out[j][i];
			min_out[i] = min_out[i] < data_out[j][i] ? min_out[i] : data_out[j][i];
		}
	}

	for (i = 0; i < IN; i++) {
		for (j = 0; j < DATA; j++)
			data_in[j][i] = (data_in[j][i] - min_in[i] + 1) / (max_in[i] - min_in[i] + 1);
	}

	for (i = 0; i < OUT; i++) {
		for (j = 0; j < DATA; j++)
			data_out[j][i] = (data_out[j][i] - min_out[i] + 1) / (max_out[i] - min_out[i] + 1);
	}

	for (i = 0; i < NEURON; i++) {	
		for (j = 0; j < IN; j++) {	
			input_weight[i][j] = rand() * 2.0 / RAND_MAX - 1;
			input_delta[i][j] = 0;
		}
	}

	for (i = 0; i < OUT; i++) {
		for (j = 0; j < NEURON; j++) {
			output_weight[i][j] = rand() * 2.0 / RAND_MAX - 1;
			output_delta[i][j] = 0;
		}
	}
}

/* 计算输出 */
void comput_output(int var) {

	int i,j;
	double sum;
	for (i = 0; i < NEURON; i++) {
		sum = 0;
		for (j = 0; j < IN; j++) {
			sum += input_weight[i][j] * data_in[var][j];
		}
		activate[i] = 1 / (1 + exp(-1 * sum));
	}

	for (i = 0; i < OUT; i++) {
		sum = 0;
		for (j = 0; j < NEURON; j++) {
			sum += output_weight[i][j] * activate[j];
		}
		output_data[i]= sum;
	}
}

/* 反馈学习 */
void back_update(int var) {

	int i, j;
	double tmp;
	for (i = 0; i < NEURON; i++) {
		tmp = 0;
		for (j = 0; j < OUT; j++) {
			tmp += (output_data[j] - data_out[var][j]) * output_weight[j][i];

			output_delta[j][i] = LEARN * output_delta[j][i] + LEARN * (output_data[j]-data_out[var][j]) * activate[i];
			output_weight[j][i] -= output_delta[j][i];
		}

		for (j = 0; j < IN; j++) {
			input_delta[i][j] = LEARN * input_delta[i][j] + LEARN * tmp * activate[i] * (1-activate[i]) * data_in[var][j];
			input_weight[i][j] -= input_delta[i][j];
		}
	}
}

/*double result(double var1,double var2) {

	int i,j;
	double sum;

	var1 = (var1 - min_in[0] + 1) / (max_in[0] - min_in[0] + 1);
	var2 = (var2 - min_in[1] + 1) / (max_in[1] - min_in[1] + 1);

	for (i = 0; i < NEURON; i++) {
		sum = 0;
		sum = input_weight[i][0] * var1 + input_weight[i][1] * var2;
		activate[i] = 1 / (1 + exp(-1 * sum));
	}
	sum = 0;
	for (j = 0; j < NEURON; j++)
		sum += output_weight[0][j] * activate[j];

	return sum * (max_out[0] - min_out[0] + 1) + min_out[0] - 1;
}*/

/* 将训练后的权值写入到文件中 */
void write_neuron() {

	int i, j;
	FILE *fp;
	if ((fp = fopen(NEURON_WEIGHT, "w")) == NULL) {
		fprintf(stderr, "can not open the neuron file\n");
		exit(EXIT_FAILURE);
	}

	for (i = 0; i < NEURON; i++) {	
		for (j = 0; j < IN; j++) {
			fprintf(fp, "%lf ", input_weight[i][j]);
		}
		fprintf(fp, "\n");
	}
	fprintf(fp, "\n");

	for (i = 0; i < OUT; i++) {	
		for (j = 0; j < NEURON; j++) {
			fprintf(fp, "%lf ", output_weight[i][j]);
		}
		fprintf(fp, "\n");
	}
	fclose(fp);
}

/* 从文件中读取训练好的权值 */
void read_neuron() {

	int i, j;
	FILE *fp;
	if ((fp = fopen(NEURON_WEIGHT, "r")) == NULL) {
		fprintf(stderr, "can not open the neuron file\n");
		exit(EXIT_FAILURE);
	}

	for (i = 0; i < NEURON; i++) {	
		for (j = 0; j < IN; j++) {
			fscanf(fp, "%lf", &input_weight[i][j]);
		}
	}
	for (i = 0; i < OUT; i++) {	
		for (j = 0; j < NEURON; j++) {
			fscanf(fp, "%lf", &output_weight[i][j]);
		}
	}
	fclose(fp);
}

/* 训练神经网络 */
void  train_network() {

	int i, j, time = 0;
	do {
		error = 0.0;
		for (i = 0; i < DATA; i++) {
			comput_output(i);
			for (j = 0; j < OUT; j++)
				error += fabs((output_data[j] - data_out[i][j]) / data_out[i][j]);
			back_update(i);
		}
		time++;
		printf("%d  %lf\n",time, error / DATA);
	} while (time < TRAINC && error / DATA > ERROR);
}

/* 输出权值，用于调试 */
void print_weight() {

	int i, j;
	for (i = 0; i < NEURON; i++) {	
		for (j = 0; j < IN; j++) {
			printf("%lf ", input_weight[i][j]);
		}
		printf("\n");
	}
	printf("\n");

	for (i = 0; i < OUT; i++) {	
		for (j = 0; j < NEURON; j++) {
			printf("%lf ", output_weight[i][j]);
		}
		printf("\n");
	}
}

int main(int argc, char *argv[]) {
	read_data();
	init_bpnetwork();
	train_network();
	//write_neuron();
	//read_neuron();
	return 0;
}
