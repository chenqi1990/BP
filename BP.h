// BP.h: interface for the BP class.
//
//////////////////////////////////////////////////////////////////////

#define IN 1024
#define HIDE 60
#define OUT 2

#define E_MIN 0.001
#define MAX_TIMES 5000//150000
#define STEP 0.001

#define TRAIN_PATH "traindata.txt"
#define TEST_PATH "testdata.txt"
#define MODOL_PATH "modol.dat"

#include <math.h>

#include <fstream>
#include <map>
#include <list> 
#include <string>
#include <iostream>
#include <ctime>
using namespace std;

#pragma once

//ѵ������
class Material{
public:
	int correct;
    double data[IN];
};
/*ʹ�÷�����
	ѵ����
		�ȵ���getTrainData����ȡ��ѵ������
		�ٵ���train����ѵ��
	ʶ��
		��Ϊin_unit��ֵ
		�ٵ���classifyֱ��ʶ��
*/
//BPϵͳ��
class BP  
{
public:
	int predict();
	int readWeights();
	void saveWeights();
	void getInput_Output(Material &train_data);
	void train(string test_path);
	double test(string path);
	BP();
	virtual ~BP();
	void initial();
	int getTrainData( string path);
	void visualize();

	
	list<Material> train_data;//�洢����ѵ�����мල����
	int data_num;

	double max_min_input[IN][2];
	double max_min_output[OUT][2];
	double in_unit[IN+1];//���һ����Ϊ��ֵ��
	double hid_unit[HIDE+1];//���һ����Ϊ��ֵ��
	double out_unit[OUT];
	double target_out[OUT];

	double wih[HIDE][IN+1];
	double who[OUT][HIDE+1];
	double delta_wih[HIDE][IN+1];
	double delta_who[OUT][HIDE+1];

	double alpha;

};
