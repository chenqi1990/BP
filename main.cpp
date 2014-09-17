#include "BP.h"
#include <iostream>
#include <string>
using namespace std;
#define TRAIN 1
#define TEST 2
#define EXIT 0
#define CONVERT 3
#define VISUALIZE 4

void Convert(const string& pos_list, const string& neg_list, const string& output_file);
int Choice(){
    int choice;
    cout<<"******************************************"<<endl
        <<"* train                                1 *"<<endl
        <<"* test                                 2 *"<<endl
        <<"* convert images to usable data format 3 *"<<endl
		<<"* visualize                            4 *"<<endl
        <<"* exit                                 0 *"<<endl
        <<"*****************************************"<<endl;
    cin>>choice;

    return choice;
}
int main(int argc, char **argv)
{
    BP bp;
    int choice;
	string neg_path, pos_path, train_path, test_path, model_path;

	model_path = "model.dat";
	test_path = "test_data.dat";
	train_path = "train_data.dat";

    while( (choice=Choice()) != 0 ){
		switch( choice ){
		case TRAIN:
            if( bp.getTrainData( train_path ) > 0 ){
                bp.train(test_path);
            }else{
                cout<<"训练数据读入出错\n";
            }
            break;
        case TEST:
            if(	bp.readWeights() == 1 ){

                cout<<"Accuracy : "<< bp.test(test_path) <<endl;
            }else{
                cout<<"模型数据读入出错\n";
            }
            break;
        case CONVERT:
			neg_path = "pointing_neg_train.list";
			pos_path = "pointing_pos_train.list";
			cout<<train_path<<endl;
			Convert(pos_path, neg_path, train_path);

			neg_path = "pointing_neg_test.list";
			pos_path = "pointing_pos_test.list";
			cout<<test_path<<endl;
			Convert(pos_path, neg_path, test_path);
			break;
		case VISUALIZE:
			if(	bp.readWeights() == 1 ){
				bp.visualize();
			}else{
				cout<<"模型数据读入出错\n";
			}
			break;
        default:
            break;
        }
    }
    return 0;
}
