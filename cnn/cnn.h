#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <unistd.h>
#include <string.h>
#include <iostream>
#include <fstream>
using namespace std;
struct Featuremap{
    int w;
    int h;
    int ch;
    double *fm_ptr;
};
struct Kernel{
    int kernel_w;
    int kernel_h;
    int stride;
};
struct Weight{
    double *wgt_ptr;
    double *bias_ptr;
    int wgt_size;
    int bias_size;
};


void deletefm(Featuremap fm);
void deletewgt(Weight wgt);
Featuremap createfm(int w,int h,int ch);
Weight createwgt(int kw,int kh,int ich,int och);
Kernel createkernel(int kw,int kh,int stride);

void conv2d(Featuremap *ifm,Kernel *ck,Weight *wgt,
            bool relu_en, Featuremap *ofm);
void maxpool(Featuremap *ifm,Kernel *pk, Featuremap *ofm);
void fc(Featuremap *ifm,Weight *wgt,bool relu_en,Featuremap *ofm);
void softmax(Featuremap *ifm,Featuremap *ofm);
void load_wgt(char * filename,Weight *wgt,bool is_bias);
void printfm(Featuremap *fm);
