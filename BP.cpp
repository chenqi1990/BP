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
�ú�����Ҫ������Ȩֵ�������Сֵ -0.05~0.05
*/
void BP::initial(){
	int i,j;
	srand( (unsigned int)time(0) );
	//��ʼ������㵽���ز��Ȩֵ
	for(i=0; i<HIDE; i++){
		for(j=0; j<IN+1; j++){
			//wih[i][j] = (double)(((double)rand()/RAND_MAX) - 0.5)/10;
			wih[i][j] = (double)(((double)rand()/RAND_MAX) - 0.5)/10;
		}
	}
	//��ʼ�����ز㵽������Ȩֵ
	for(i=0; i<OUT; i++){
		for(j=0; j<HIDE+1; j++){
			who[i][j] = (double)(((double)rand()/RAND_MAX) - 0.5)/10;
		}
	}
	alpha = STEP;
}
/*
�ú������ڻ�ȡѵ������Ĳ���
���ػ�ȡ�Ĳ��ϸ���
*/
int BP::getTrainData(string path){
	Material material;
	int total_num, i;
	float background = 0;
	int count;
	
	//1 ��ѵ�������ļ�TRAIN_PATH
	ifstream input( path, ios::in|ios::binary );
	if(!input){
		cout<<"********��ѵ���ļ�ʧ��*********"<<endl;
		return -1;
	}
	//2 ����ѵ����������
	input.read((char*)(&total_num), sizeof(int));
	input.read((char*)(&count), sizeof(int));
	if (count != IN){
		cout<<"����ά������\n";
		return -1;
	}
	//3 �ֱ����ÿ��ѵ������
	this->train_data.clear();
	for(i=0; i<total_num; i++){
		//3.1 �������ѵ�����ݵ�����ֵ
		input.read( (char*)(material.data), sizeof(double)*count );
		//3.2 �������ѵ�����ݵ���ʵֵ
		input.read( (char*)(&(material.correct)) , sizeof(int) );
		//3.3 ���������ݴ���ѵ������
		this->train_data.push_back(material);
	}
	input.close();
	this->data_num = total_num;
	return total_num;
}
/*
��ѵ�����ϸ�ֵ��ģ�͵����뵥Ԫ
���Ҹ�ֵĿ���������
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
	this->in_unit[i] = 1;//��ֵ��
	memset((void*)target_out, 0, sizeof(double)*OUT);
	target_out[ train_data.correct ] = 0.99;
// 	for(i=0; i<OUT; i++){
// 		this->target_out[i] = train_data.correct;
// 		//this->target_out[i] = 1.0/( 1.0 + exp( -(double)(train_data.correct) ) );
// 		//this->target_out[i] = ( train_data.correct - max_min_output[i][0] ) / ( max_min_output[i][1]- max_min_output[i][0] );
// 	}
}

/*
BP�㷨��ѵ������
*/
void BP::train(string test_path)
{
	this->initial();//��ʼ��BPģ��

	int train_times;
	int i,j;
	double unit_in,temp;
	//���
	double error_out[OUT];
	double error_hid[HIDE+1];
	double error_sum;
	//��ʼѵ��
	int count_instance = 0;
	train_times = 0;
	list<Material>::iterator it;
	srand( (int)time(0) );
	do{
		error_sum = 0;
		memset((void*)(delta_who), 0, sizeof(double)*OUT*(HIDE+1));
		memset((void*)(delta_wih), 0, sizeof(double)*HIDE*(IN+1));
		for(it=train_data.begin(); it!=train_data.end(); it++, count_instance++){//��ÿ��һ��ѵ������
			if (it->correct == 1){
				if (rand()%4 != 0)
					continue;
			}
			//�ӵ�ǰʵ����ȡ�����ȡ��׼���
			getInput_Output(*it);
			//ǰ����㣬��ȡÿ����Ԫ�����ֵ
			//���ص�Ԫ
			//#pragma omp parallel for
			for(i=0; i<HIDE; i++){
				unit_in = 0;
				for(j=0; j<IN+1; j++){
					unit_in += wih[i][j] * in_unit[j];
				}
				hid_unit[i] = 1.0/( 1.0 + exp(-unit_in) );
			}
			hid_unit[HIDE] = 1;//��ֵ��
			//�����Ԫ
			//#pragma omp parallel for
			for(i=0; i<OUT; i++){
				unit_in = 0;
				for(j=0; j<HIDE+1; j++){
					unit_in += who[i][j] * hid_unit[j];
				}
				out_unit[i] = 1.0/( 1.0 + exp(-unit_in) );
			}
			//�������ÿ�������Ԫ�������������
			for(i=0; i<OUT; i++){
				error_out[i] = out_unit[i]*(1-out_unit[i])*(target_out[i] - out_unit[i]);
				error_sum += fabs( error_out[i] );
			}
			//�������ÿ�����ص�Ԫ�������������
			for(i=0; i<HIDE+1; i++){
				error_hid[i] = hid_unit[i]*(1-hid_unit[i]);
				//��ǰ���ص�Ԫ�����������Ԫ�����Ĺ���
				temp = 0;
				for(j=0; j<OUT; j++){
					temp += who[j][i] * error_out[j];
				}
				error_hid[i] *= temp;
				error_sum += fabs( error_hid[i] );
			}
			//����ÿ������Ȩֵ-------------------------------------------------------------
			//���ز㵽�����Ȩֵ
			for(i=0; i<OUT; i++){
				for(j=0; j<HIDE+1; j++){
					delta_who[i][j] += alpha * error_out[i] * hid_unit[j];
				}
			}
			//����㵽���ز�
			for(i=0; i<HIDE; i++){
				for(j=0; j<IN+1; j++){
					delta_wih[i][j] += alpha * error_hid[i] * in_unit[j];
				}
			}
			if (count_instance%50 == 0){
				//���ز㵽�����Ȩֵ
				for(i=0; i<OUT; i++){
					for(j=0; j<HIDE+1; j++){
						who[i][j] += delta_who[i][j];
					}
				}
				//����㵽���ز�
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
		//�жϵ�ǰ״̬�Ƿ��������
		//����error_sum�͵����������ж�
		if(train_times > MAX_TIMES)
			break;
		if(error_sum <= E_MIN)
			break;
	}while(true);

	cout<<"ѵ�����"<<endl;
	//����Ȩֵ
	saveWeights();
}
/*
�ú�����������Ȩֵ���ļ���
*/
void BP::saveWeights()
{
	ofstream out( MODOL_PATH, ios_base::binary | ios_base::out);
	if(!out){
		cout<<"����ģ��ʧ��"<<endl;
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
�ú����������ļ��ж�ȡȨֵ
*/
int BP::readWeights()
{
 	ifstream input( MODOL_PATH, ios_base::in | ios_base::binary);
	if(!input){
		cout<<"��ģ���ļ�ʧ��"<<endl;
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
�ú�������Ԥ�⵱ǰ��ʵ��
����Ԥ��Ľ��
ǰ�᣺�Ѿ������뵥Ԫ��ֵ����
*/
int BP::predict()
{
	int i,j;
	double unit_in;
	//ǰ����㣬��ȡÿ����Ԫ�����ֵ
	//���ص�Ԫ
	for(i=0; i<HIDE; i++){
		unit_in = 0;
		for(j=0; j<IN+1; j++){
			unit_in += wih[i][j] * in_unit[j];
		}
		hid_unit[i] = 1.0/( 1.0 + exp(-unit_in) );
	}
	hid_unit[i] = 1;//��ֵ��
	//�����Ԫ
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

	//1 �򿪲��������ļ�
	ifstream input( path, ios::in|ios::binary );
	if(!input){
		cout<<"********�򿪲����ļ�ʧ��*********"<<endl;
		return -1;
	}
	//2 ���������������
	input.read((char*)(&total_num), sizeof(int));
	input.read((char*)(&count), sizeof(int));
	if (count != IN){
		cout<<"����ά������\n";
		return -1;
	}
	//3 �ֱ����ÿ���������
	int pos_count = 0, neg_count = 0;
	int pos_cor = 0, neg_cor = 0;
	for(i=0; i<total_num; i++){
		//3.1 �������������ݵ�����ֵ
		input.read( (char*)(in_unit), sizeof(double)*count );
		in_unit[count] = 1;//��ֵ��
		//3.2 �������������ݵ���ʵֵ
		input.read( (char*)(&(correct)) , sizeof(int) );

		get = predict();
		//get = log( get/(1-get) );
		//get = get*(max_min_output[0][0] - max_min_output[0][1]) + max_min_output[0][1];
		//3.3 ����Ƚ�
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