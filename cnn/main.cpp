#include <stdio.h>
#include <stdlib.h>
#include <opencv2/opencv.hpp>
#include "cnn.h"
using namespace cv;
using namespace std;

void fill_testimg(Featuremap *fm);
int main(int argc, char** argv )
{
    
  if ( argc != 2 )
  {
    printf("usage: ./main <Image_Path>\n");
    return -1;
  }

  Mat ori_img;
  ori_img = imread( argv[1], 1 );
  if ( !ori_img.data ){
        printf("No image data. Exit\n\n");
        return -1;
  }
    printf("input img [%s]:row=%d,col=%d,ch=%d \n",argv[1],ori_img.rows,ori_img.cols,ori_img.channels());
  
  //1.Build the cnn network
  Featuremap fm_in=createfm(28,28,1);
  
    
  Kernel k_conv1=createkernel(3,3,1);//(int kw,int kh,int stride);
  Weight w_conv1=createwgt(3,3,1,32);//(int kw,int kh,int ich,int och);
  Featuremap fm_conv1=createfm(28,28,32);//(int w,int h,int ch);
    
  Kernel k_pool1=createkernel(2,2,2);//(int kw,int kh,int stride);
  Featuremap fm_pool1=createfm(14,14,32);//(int w,int h,int ch);
    
  Kernel k_conv2=createkernel(3,3,1);//(int kw,int kh,int stride);
  Weight w_conv2=createwgt(3,3,32,64);//(int kw,int kh,int ich,int och);
  Featuremap fm_conv2=createfm(14,14,64);//(int w,int h,int ch);
    
  Kernel k_pool2=createkernel(2,2,2);//(int kw,int kh,int stride);
  Featuremap fm_pool2=createfm(7,7,64);//(int w,int h,int ch);
    
  Weight w_fc1=createwgt(7,7,64,64);//(int kw,int kh,int ich,int och);
  Featuremap fm_fc1=createfm(1,1,64);//(int w,int h,int ch);
    
  Weight w_fc2=createwgt(1,1,64,10);//(int kw,int kh,int ich,int och);
  Featuremap fm_fc2=createfm(1,1,10);//(int w,int h,int ch);
    
  Featuremap fm_softmax=createfm(1,1,10);//(int w,int h,int ch);
  
  //2.Initialize the weights and input fm
  int i,j;
  for(i=0;i<28*28;i++){
      *(fm_in.fm_ptr+i)=(double)*(ori_img.data+i*ori_img.channels());
      *(fm_in.fm_ptr+i)/=255;
  }
  //fill_testimg(&fm_in);
    
  load_wgt((char *)"../wgt/conv1_w.txt",&w_conv1,false);
  load_wgt((char *)"../wgt/conv1_b.txt",&w_conv1,true);
  load_wgt((char *)"../wgt/conv2_w.txt",&w_conv2,false);
  load_wgt((char *)"../wgt/conv2_b.txt",&w_conv2,true);
  load_wgt((char *)"../wgt/fc1_w.txt",&w_fc1,false);
  load_wgt((char *)"../wgt/fc1_b.txt",&w_fc1,true);
  load_wgt((char *)"../wgt/fc2_w.txt",&w_fc2,false);
  load_wgt((char *)"../wgt/fc2_b.txt",&w_fc2,true);
    
  //3.Forward
  printf("\n");
  conv2d(&fm_in,&k_conv1,&w_conv1,true,&fm_conv1);//(Featuremap *ifm,Kernel *ck,Weight *wgt,bool relu_en, Featuremap *ofm)
  maxpool(&fm_conv1,&k_pool1,&fm_pool1);//(Featuremap *ifm,Kernel *pk, Featuremap *ofm)
  conv2d(&fm_pool1,&k_conv2,&w_conv2,true,&fm_conv2);//(Featuremap *ifm,Kernel *ck,Weight *wgt,bool relu_en, Featuremap *ofm)
  maxpool(&fm_conv2,&k_pool2,&fm_pool2);//(Featuremap *ifm,Kernel *pk, Featuremap *ofm)
  fc(&fm_pool2,&w_fc1,true,&fm_fc1);//(Featuremap *ifm,Weight *wgt,bool relu_en,Featuremap *ofm)
  fc(&fm_fc1,&w_fc2,false,&fm_fc2);//(Featuremap *ifm,Weight *wgt,bool relu_en,Featuremap *ofm)
  softmax(&fm_fc2,&fm_softmax);
    
  //printfm(&fm_in);
  //printfm(&fm_conv1);
  //printfm(&fm_pool1);printf("\n");
  //printfm(&fm_conv2);
  //printfm(&fm_pool2);
  //printfm(&fm_fc1);
  //printfm(&fm_fc2);
  //printfm(&fm_softmax);
    
  //4. free mem
  printf("\n");
  deletefm(fm_pool1);
  deletefm(fm_conv2);
  deletefm(fm_pool2);
  deletefm(fm_fc1);
  deletefm(fm_fc2);
  deletewgt(w_conv1);
  deletewgt(w_conv2);
  deletewgt(w_fc1);
  deletewgt(w_fc2);
    
  //5.Display the results:
  int sel_i=-1;
  double t=-1;
  printf("\n");
  for(i=0;i<10;i++){
      printf("Prob%d:%.2lf\n",i,*(fm_softmax.fm_ptr+i));
      if(t<*(fm_softmax.fm_ptr+i)){
            sel_i=i;
            t=*(fm_softmax.fm_ptr+i);
        }
    }
  deletefm(fm_softmax);
  printf("\n[Final Result]: This pic is Num %d\n\n",sel_i);
    
  //6.opencv display(20x zoom)
  int zoom=20;
  Mat image(28*zoom,28*zoom,CV_8UC(1));
  for(i=0;i<28*zoom;i++){
      for(j=0;j<28*zoom;j++){
        *(image.data+i+j*28*zoom)=(unsigned char)(*(fm_in.fm_ptr+(i/zoom)+(j/zoom)*28)*255);
      }
  }
  deletefm(fm_in);
  //printf("row=%d,col=%d,ch=%d \n",image.rows,image.cols,image.channels());
  //cout << "M = " << endl << " " << image << endl << endl;
  namedWindow("Display Image", WINDOW_AUTOSIZE );
  imshow("Display Image", image);
  waitKey(0);
    
  return 0;
}


void fill_testimg(Featuremap *fm){

    double testimg[]={0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.172549024,0.498039246,0.945098102,0.839215755,0.662745118,0.627451,0.0784313753,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.0627451,0.90196085,0.996078491,0.996078491,0.749019623,0.937254965,0.996078491,0.788235366,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.745098054,0.996078491,0.996078491,0.427451,0.24313727,0.741176486,0.996078491,0.964705944,0.160784319,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.882353,0.996078491,0.996078491,0.956862807,0.988235354,0.996078491,0.996078491,0.996078491,0.254901975,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.541176498,0.964705944,0.996078491,0.996078491,0.627451,0.407843173,0.866666734,0.996078491,0.254901975,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.137254909,0.423529446,0.141176477,0.0156862754,0,0.78039223,0.996078491,0.254901975,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.0705882385,0.90196085,0.996078491,0.254901975,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.192156881,0.996078491,0.952941239,0.117647067,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.635294139,0.996078491,0.850980461,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.352941185,0.509803951,0.509803951,0.509803951,0.192156881,0,0.0235294141,0.831372619,0.996078491,0.403921604,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.266666681,0.984313786,0.996078491,0.996078491,0.996078491,0.968627512,0.549019635,0.674509823,0.996078491,0.94901967,0.121568635,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.592156887,0.996078491,0.380392194,0.184313729,0.541176498,0.976470649,0.996078491,0.996078491,0.996078491,0.345098048,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.552941203,0.996078491,0.254901975,0,0,0.501960814,0.996078491,0.996078491,0.996078491,0.21960786,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.231372565,0.988235354,0.588235319,0,0.333333343,0.964705944,0.996078491,0.996078491,0.996078491,0.21960786,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.917647123,0.945098102,0.647058845,0.968627512,0.996078491,0.760784388,0.776470661,0.996078491,0.478431404,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.752941251,0.996078491,1,0.984313786,0.58431375,0.0431372561,0.478431404,0.996078491,0.694117665,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.0666666701,0.345098048,0.53725493,0.21960786,0,0,0.478431404,0.996078491,0.894117713,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.6,0.996078491,0.894117713,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.815686345,1,0.776470661,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.698039234,0.996078491,0.329411775,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};
    memcpy((*fm).fm_ptr,testimg,(*fm).w*(*fm).h*(*fm).ch*sizeof(double));
}
