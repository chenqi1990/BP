// BP.cpp: implementation of the BP class.
//
//////////////////////////////////////////////////////////////////////
#include "BP.h"
#include <time.h>
#include <math.h>
#include <omp.h>

#include <opencv/cv.h>
#include <opencv/cxcore.h>
#include <opencv/highgui.h>

BP::BP()
{
	//this->readWeights();
 	max_min_input[0][0] = 5.1;
	max_min_input[0][1] = 0.9;
	max_min_output[0][0] = 10.1;
	max_min_output[0][1] = 1.9;
}

BP::~BP()
{

}
/*
该函数主要用来给权值分配随机小值 -0.05~0.05
*/
void BP::initial(){
	int i,j;
	srand( (unsigned int)time(0) );
	//初始化输入层到隐藏层的权值
	for(i=0; i<HIDE; i++){
		for(j=0; j<IN+1; j++){
			//wih[i][j] = (double)(((double)rand()/RAND_MAX) - 0.5)/10;
			wih[i][j] = (double)(((double)rand()/RAND_MAX) - 0.5)/10;
		}
	}
	//初始化隐藏层到输出层的权值
	for(i=0; i<OUT; i++){
		for(j=0; j<HIDE+1; j++){
			who[i][j] = (double)(((double)rand()/RAND_MAX) - 0.5)/10;
		}
	}
	alpha = STEP;
}
/*
该函数用于获取训练所需的材料
返回获取的材料个数
*/
int BP::getTrainData(string path){
	Material material;
	int total_num, i;
	float background = 0;
	int count;
	
	//1 打开训练数据文件TRAIN_PATH
	ifstream input( path, ios::in|ios::binary );
	if(!input){
		cout<<"********打开训练文件失败*********"<<endl;
		return -1;
	}
	//2 读入训练数据总数
	input.read((char*)(&total_num), sizeof(int));
	input.read((char*)(&count), sizeof(int));
	if (count != IN){
		cout<<"数据维数错误\n";
		return -1;
	}
	//3 分别读入每组训练数据
	this->train_data.clear();
	for(i=0; i<total_num; i++){
		//3.1 读入该组训练数据的属性值
		input.read( (char*)(material.data), sizeof(double)*count );
		//3.2 读入该组训练数据的真实值
		input.read( (char*)(&(material.correct)) , sizeof(int) );
		//3.3 将该组数据存入训练集中
		this->train_data.push_back(material);
	}
	input.close();
	this->data_num = total_num;
	return total_num;
}
/*
将训练材料赋值给模型的输入单元
并且赋值目标输出向量
*/
void BP::getInput_Output(Material &train_data)
{
	int i;
	for(i=0; i<IN; i++){
		this->in_unit[i] = train_data.data[i];
		//this->in_unit[i] = 1.0/( 1.0 + exp( -train_data.data[i]) );
		//this->in_unit[i] = ( train_data.data[i] - max_min_input[i][0] ) / ( max_min_input[i][1]- max_min_input[i][0] );
	}
	//this->in_unit[0] /= 100;
	this->in_unit[i] = 1;//阈值项
	memset((void*)target_out, 0, sizeof(double)*OUT);
	target_out[ train_data.correct ] = 0.99;
// 	for(i=0; i<OUT; i++){
// 		this->target_out[i] = train_data.correct;
// 		//this->target_out[i] = 1.0/( 1.0 + exp( -(double)(train_data.correct) ) );
// 		//this->target_out[i] = ( train_data.correct - max_min_output[i][0] ) / ( max_min_output[i][1]- max_min_output[i][0] );
// 	}
}

/*
BP算法的训练函数
*/
void BP::train(string test_path)
{
	this->initial();//初始化BP模型

	int train_times;
	int i,j;
	double unit_in,temp;
	//误差
	double error_out[OUT];
	double error_hid[HIDE+1];
	double error_sum;
	//开始训练
	int count_instance = 0;
	train_times = 0;
	list<Material>::iterator it;
	srand( (int)time(0) );
	do{
		error_sum = 0;
		memset((void*)(delta_who), 0, sizeof(double)*OUT*(HIDE+1));
		memset((void*)(delta_wih), 0, sizeof(double)*HIDE*(IN+1));
		for(it=train_data.begin(); it!=train_data.end(); it++, count_instance++){//对每个一个训练样例
			if (it->correct == 1){
				if (rand()%4 != 0)
					continue;
			}
			//从当前实例获取输入和取标准输出
			getInput_Output(*it);
			//前向计算，获取每个单元的输出值
			//隐藏单元
			//#pragma omp parallel for
			for(i=0; i<HIDE; i++){
				unit_in = 0;
				for(j=0; j<IN+1; j++){
					unit_in += wih[i][j] * in_unit[j];
				}
				hid_unit[i] = 1.0/( 1.0 + exp(-unit_in) );
			}
			hid_unit[HIDE] = 1;//阈值项
			//输出单元
			//#pragma omp parallel for
			for(i=0; i<OUT; i++){
				unit_in = 0;
				for(j=0; j<HIDE+1; j++){
					unit_in += who[i][j] * hid_unit[j];
				}
				out_unit[i] = 1.0/( 1.0 + exp(-unit_in) );
			}
			//对网络的每个输出单元，计算其误差项
			for(i=0; i<OUT; i++){
				error_out[i] = out_unit[i]*(1-out_unit[i])*(target_out[i] - out_unit[i]);
				error_sum += fabs( error_out[i] );
			}
			//对网络的每个隐藏单元，计算其误差项
			for(i=0; i<HIDE+1; i++){
				error_hid[i] = hid_unit[i]*(1-hid_unit[i]);
				//当前隐藏单元对所有输出单元的误差的贡献
				temp = 0;
				for(j=0; j<OUT; j++){
					temp += who[j][i] * error_out[j];
				}
				error_hid[i] *= temp;
				error_sum += fabs( error_hid[i] );
			}
			//更新每个网络权值-------------------------------------------------------------
			//隐藏层到输出层权值
			for(i=0; i<OUT; i++){
				for(j=0; j<HIDE+1; j++){
					delta_who[i][j] += alpha * error_out[i] * hid_unit[j];
				}
			}
			//输入层到隐藏层
			for(i=0; i<HIDE; i++){
				for(j=0; j<IN+1; j++){
					delta_wih[i][j] += alpha * error_hid[i] * in_unit[j];
				}
			}
			if (count_instance%50 == 0){
				//隐藏层到输出层权值
				for(i=0; i<OUT; i++){
					for(j=0; j<HIDE+1; j++){
						who[i][j] += delta_who[i][j];
					}
				}
				//输入层到隐藏层
				for(i=0; i<HIDE; i++){
					for(j=0; j<IN+1; j++){
						wih[i][j] += delta_wih[i][j];
					}
				}
				memset((void*)(delta_who), 0, sizeof(double)*OUT*(HIDE+1));
				memset((void*)(delta_wih), 0, sizeof(double)*HIDE*(IN+1));
			}
		}
		if (train_times%500 == 0)
			alpha *= 0.9;
		train_times++;
		if (train_times%100 == 0){
			saveWeights();
		}
		if (train_times%20 == 0){
			//readWeights();
			double accuracy = test(test_path);

			cout<<train_times<<"\t"<<alpha<<"\t"<<error_sum<<"\t"<<accuracy<<endl;
			cout<<"----------------------------------------------------------------\n";
			//log<<train_times<<"\t"<<alpha<<"\t"<<error_sum<<"\t"<<accuracy<<endl;
		}
		//判断当前状态是否符合条件
		//根据error_sum和迭代次数来判断
		if(train_times > MAX_TIMES)
			break;
		if(error_sum <= E_MIN)
			break;
	}while(true);

	cout<<"训练完毕"<<endl;
	//保存权值
	saveWeights();
}
/*
该函数用来保存权值到文件中
*/
void BP::saveWeights()
{
	ofstream out( MODOL_PATH, ios_base::binary | ios_base::out);
	if(!out){
		cout<<"保存模型失败"<<endl;
		return;
	}
//	ofstream out("weight.data");
	int i,j;
	double temp;

	for(i=0; i<HIDE; i++){
		for(j=0; j<IN+1; j++){
			temp = wih[i][j];
			out.write( (char*)(&temp), sizeof(double) );
			//out<<wih[i][j]<<" "<<endl;
			//out<<wih[i][j];
		}
	}
	for(i=0; i<OUT; i++){
		for(j=0; j<HIDE+1; j++){
			temp = who[i][j];
			out.write( (char*)(&temp), sizeof(double) );
			//out<<who[i][j];
		}
	}

	out.close();
}
/*
该函数用来从文件中读取权值
*/
int BP::readWeights()
{
 	ifstream input( MODOL_PATH, ios_base::in | ios_base::binary);
	if(!input){
		cout<<"打开模型文件失败"<<endl;
		return -1;
	}
	//ifstream in("weight.data");
	int i,j;
	double temp;
	
	for(i=0; i<HIDE; i++){
		for(j=0; j<IN+1; j++){
			input.read( (char*)(&temp), sizeof(double) );
			wih[i][j] = temp;
			//in>>wih[i][j];
		}
	}
	for(i=0; i<OUT; i++){
		for(j=0; j<HIDE+1; j++){
			input.read( (char*)(&temp), sizeof(double) );
			who[i][j] = temp;
			//input>>who[i][j];
		}
	}

	input.close();
	return 1;
}
/*
该函数用来预测当前的实例
返回预测的结果
前提：已经把输入单元赋值好了
*/
int BP::predict()
{
	int i,j;
	double unit_in;
	//前向计算，获取每个单元的输出值
	//隐藏单元
	for(i=0; i<HIDE; i++){
		unit_in = 0;
		for(j=0; j<IN+1; j++){
			unit_in += wih[i][j] * in_unit[j];
		}
		hid_unit[i] = 1.0/( 1.0 + exp(-unit_in) );
	}
	hid_unit[i] = 1;//阈值项
	//输出单元
	int max;
	double max_out = -100;
	for(i=0; i<OUT; i++){
		unit_in = 0;
		for(j=0; j<HIDE+1; j++){
			unit_in += who[i][j] * hid_unit[j];
		}
		out_unit[i] = 1.0/( 1.0 + exp(-unit_in) );
		if (out_unit[i] > max_out ){
			max_out = out_unit[i];
			max = i;
		}
	}
	return max;
}

double BP::test(string path){
	int total_num, i;
	int correct = 0;
	int correct_num = 0;
	double get;
	int count;

	//1 打开测试数据文件
	ifstream input( path, ios::in|ios::binary );
	if(!input){
		cout<<"********打开测试文件失败*********"<<endl;
		return -1;
	}
	//2 读入测试数据总数
	input.read((char*)(&total_num), sizeof(int));
	input.read((char*)(&count), sizeof(int));
	if (count != IN){
		cout<<"数据维数错误\n";
		return -1;
	}
	//3 分别读入每组测试数据
	int pos_count = 0, neg_count = 0;
	int pos_cor = 0, neg_cor = 0;
	for(i=0; i<total_num; i++){
		//3.1 读入该组测试数据的属性值
		input.read( (char*)(in_unit), sizeof(double)*count );
		in_unit[count] = 1;//阈值项
		//3.2 读入该组测试数据的真实值
		input.read( (char*)(&(correct)) , sizeof(int) );

		get = predict();
		//get = log( get/(1-get) );
		//get = get*(max_min_output[0][0] - max_min_output[0][1]) + max_min_output[0][1];
		//3.3 输出比较
		//cout<<correct<<"\t"<<get<<"\t"<<abs(get-correct)<<endl;
		//dis += abs(get-correct);
		if (get == correct) {
			correct_num++;
			if (correct == 0) {
				pos_cor++;
				pos_count++;
			}else{
				neg_count++;
				neg_cor++;
			}
		}else{
			if (correct == 0) {
				pos_count++;
			}else{
				neg_count++;
			}
		}
	}
	input.close();
	cout<<"pos cor = "<<pos_cor<<"\tpos miss = "<<pos_count-pos_cor<<endl
		<<"neg cor = "<<neg_cor<<"\tneg miss = "<<neg_count-neg_cor<<endl;
	return (double)correct_num/total_num;
}

void BP::visualize()
{
	int i=0,j;
	IplImage* weight_img = cvCreateImage(cvSize(32,32), 8,1);
	IplImage* weight_img2 = cvCreateImage(cvSize(128,128), 8,1);
	double max_val,min_val;
	char path[100] = "";

	for (i=0; i<HIDE; i++){
		max_val = -1e100;
		min_val = 1e100;
		for (j=0; j<IN; j++){
			if ( wih[i][j] < min_val)
				min_val = wih[i][j];
			if (wih[i][j] > max_val)
				max_val = wih[i][j];
		}
		for (j=0; j<IN; j++) {
			int val = (int)(wih[i][j]-min_val)*255/(max_val-min_val);
			weight_img->imageData[j] = (char)( val );
		}
		cvResize(weight_img, weight_img2);
		//cvShowImage("weight", weight_img2);
		sprintf(path, "layer1/%f_%d.jpg",  max_val-min_val, i);
		cvSaveImage(path, weight_img2);
		//cvWaitKey();
	}
	cvReleaseImage( &weight_img );
	cvReleaseImage(&weight_img2);
}