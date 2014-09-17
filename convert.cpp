#include <fstream>
#include <string>
#include <vector>
using namespace std;
#include <opencv/cv.h>
#include <opencv/cxcore.h>
#include <opencv/highgui.h>

class Data{
public:
	string path;
	int flag;
};

#define ROWS 32
#define COLS 32

IplImage* size_normalize( IplImage* img )
{
	int width, height;
	CvRect rect;

	if( img == NULL )
		return NULL;
	width = img->width;
	height = img->height;

	if ( width < height ){
		rect.x = 0;
		rect.width = width;
		rect.y = ( height - width )/2;
		rect.height = width;

		cvSetImageROI( img, rect );
	}else if ( height < width ){
		rect.x = ( width - height )/2;
		rect.width = height;
		rect.y = 0;
		rect.height = height;

		cvSetImageROI( img, rect );
	}

	if ( width != COLS ){
		IplImage* tmp = cvCreateImage( cvSize( COLS, ROWS ), 8, img->nChannels);

		if (tmp == NULL)
			return NULL;

		cvResize( img, tmp );
		cvReleaseImage( &img );
		img = NULL;

		return tmp;
	}else{
		return img;
	}
}
bool load_image( std::string& path, const int rows, const int cols, const int channels, unsigned char* pixels )
{
	IplImage* img = NULL;
	int i,j,c;
	unsigned char* p;

	if (channels == 1)
		img = cvLoadImage( path.c_str(), 0 );
	else
		img = cvLoadImage( path.c_str() );

	if (img == NULL){
		return false;
	}
	img = size_normalize( img );
	if (img == NULL){
		return false;
	}

	for ( i=0; i<ROWS; i++){
		p = (unsigned char*)( img->imageData + img->widthStep*i );
		for ( j=0; j<COLS; j++){
			for ( c=0; c<channels; c++){
				*( pixels + ROWS*COLS*c + i*COLS + j ) = p[j*channels + c];
				//cout<<(double)p[j*channels + c]<<" ";
			}
		}
		//cout<<endl;
	}
	//cout<<endl;
	cvReleaseImage( &img );
	return true;
}

void mirror( const int rows, const int cols, const int channels, double* pixels )
{
	double* pixel_row = new double[ cols*channels ];
	int i,j,c;

	for (i=0; i<rows; i++){
		for (j=0; j<cols; j++){
			for (c=0; c<channels; c++){
				pixel_row[ cols*c + j] = pixels[rows*cols*c + cols*i + j];
			}
		}
		for (j=0; j<cols; j++){
			for (c=0; c<channels; c++)
				pixels[rows*cols*c + cols*i + j] = pixel_row[ cols*c + cols - j -1 ];
		}
	}
	delete []pixel_row;
}

//convert image data to usable data format
void Convert(const string& pos_list, const string& neg_list, const string& output_file)
{
	vector<Data> image_path;
	int pos_count, neg_count;
	ifstream pos_input( pos_list, ios::in );
	if(!pos_input){
		cout<<"********open file: "<<pos_list<<" failed*********"<<endl;
		return ;
	}
	pos_count = 0;
	while( !pos_input.eof() ){
		Data data;
		data.path = "";
		getline( pos_input, data.path );
		if (data.path == "")
			continue;
		data.flag = 0;
		image_path.push_back( data );
		pos_count++;
	}
	pos_input.close();

	ifstream neg_input( neg_list, ios::in );
	if(!neg_input){
		cout<<"********open file: "<<neg_list<<" failed*********"<<endl;
		return ;
	}
	neg_count = 0;
	while( !neg_input.eof() ){
		Data data;
		data.path = "";
		getline( neg_input, data.path );
		if (data.path == "")
			continue;
		data.flag = 1;
		image_path.push_back( data );
		neg_count++;
	}
	neg_input.close();

	random_shuffle( image_path.begin(), image_path.end() );//将image_path中的数据随机打乱

	ofstream output( output_file, ios::out|ios::binary );
	if(!output){
		cout<<"********create file: "<<output_file<<" failed*********"<<endl;
		return ;
	}
	int i, j, size = image_path.size()//+pos_count
		,count = 32*32;
	output.write(reinterpret_cast<char *>(&size),sizeof(int));
	IplImage* img;
	unsigned char* data0 = new unsigned char[count];
	output.write(reinterpret_cast<char *>(&count),sizeof(int));
	double *data = new double[count];
	int flag_;
	size = image_path.size();
	for (i=0; i<size; i++){
		//img = cvLoadImage( image_path[i].path.c_str(), 0 );
		//load_image(image_path[i].path, ROWS, COLS, 1, resized->imageData);
		if (load_image(image_path[i].path, ROWS, COLS, 1, data0) == false){
			cout<<"failed to load image : "<<image_path[i].path<<endl;
			continue;
		}
		for (j=0; j<count; j++){
			data[j] = (double)data0[j] / 255.0;
		}
		output.write( (char*)(data), sizeof(double)*count);
		flag_ = image_path[i].flag;
		output.write( (char*)(&flag_), sizeof(int) );
		//cvReleaseImage( &img );
		// 		if (flag_ == 0){
		// 			mirror( 32, 32, 1, data );
		// 			output.write( (char*)(data), sizeof(double)*count);
		// 			output.write( (char*)(&flag_), sizeof(int) );
		// 		}
	}
	delete []data0;
	delete []data;
	output.close();
}
