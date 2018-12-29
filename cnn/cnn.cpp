#include "cnn.h"
#define MAX(A,B) ((A)>(B)?(A):(B))
#define MIN(A,B) ((A)<(B)?(A):(B))
void deletefm(Featuremap fm){
    free(fm.fm_ptr);
    //printf("Free fm buf\t:ptr=0x%012lx\n",(unsigned long)fm.fm_ptr);
}
void deletewgt(Weight wgt){
    free(wgt.wgt_ptr);
    free(wgt.bias_ptr);
    //printf("Free wgt buf\t:ptr=0x%012lx\n",(unsigned long)wgt.wgt_ptr);
    //printf("Free bias buf\t:ptr=0x%012lx\n",(unsigned long)wgt.bias_ptr);
}
Featuremap createfm(int w,int h,int ch){
    Featuremap fm;
    fm.w=w;
    fm.h=h;
    fm.ch=ch;
    fm.fm_ptr=(double *)malloc(sizeof(double)*w*h*ch);
    memset(fm.fm_ptr,0,sizeof(double)*w*h*ch);
    printf("Create fm buf\t:ptr=0x%012lx,\tsize=%.2f KB,\tw/h/ch=%d/%d/%d\n",(unsigned long)fm.fm_ptr,sizeof(double)*w*h*ch/1024.0,w,h,ch);
    return fm;
}
Weight createwgt(int kw,int kh,int ich,int och){
    Weight wgt;
    wgt.wgt_size=sizeof(double)*kw*kh*ich*och;
    wgt.bias_size=sizeof(double)*och;
    wgt.wgt_ptr=(double *)malloc(wgt.wgt_size);
    wgt.bias_ptr=(double *)malloc(wgt.bias_size);
    memset(wgt.wgt_ptr,0,wgt.wgt_size);
    memset(wgt.bias_ptr,0,wgt.bias_size);
    printf("Create wgt buf\t:ptr=0x%012lx,\tsize=%.2f KB,\tkw/kh/ich/och=%d/%d/%d/%d\n",(unsigned long)wgt.wgt_ptr,wgt.wgt_size/1024.0,kw,kh,ich,och);
    printf("Create bias buf\t:ptr=0x%012lx,\tsize=%d B,\tkw/kh/ich/och=%d/%d/%d/%d\n",(unsigned long)wgt.bias_ptr,wgt.bias_size,kw,kh,ich,och);
    
    return wgt;
}
Kernel createkernel(int kw,int kh,int stride){
    Kernel kk;
    kk.kernel_w=kw;
    kk.kernel_h=kh;
    kk.stride=stride;
    return kk;
}


void conv2d(Featuremap *ifm,Kernel *ck,Weight *wgt,bool relu_en, Featuremap *ofm){
    printf("Running conv2d..\n");
    int i0,j0,k0;
    int i1,j1,k1;
    double t1,t2,w;
    int pad_lr,pad_ub;//padding="SAME"
    pad_lr=((*ck).kernel_w-(*ck).stride)/2;
    pad_ub=((*ck).kernel_h-(*ck).stride)/2;
    for(i0=0;i0<(*ofm).w;i0++){
        for(j0=0;j0<(*ofm).h;j0++){
            for(k0=0;k0<(*ofm).ch;k0++){
                t1=*((*wgt).bias_ptr+k0);
                int j1_st=MAX((*ck).stride*j0-pad_ub,0);
                int j1_end=MIN((*ifm).h,(*ck).stride*j0-pad_ub+(*ck).kernel_h);
                int i1_st=MAX((*ck).stride*i0-pad_lr,0);
                int i1_end=MIN((*ifm).w,(*ck).stride*i0-pad_lr+(*ck).kernel_w);
                for(k1=0;k1<(*ifm).ch;k1++){
                   for(j1=j1_st;j1<j1_end;j1++){
                       for(i1=i1_st;i1<i1_end;i1++){
                           t2=*((*ifm).fm_ptr+i1+j1*(*ifm).w+k1*(*ifm).w*(*ifm).h);//+0.1;
                            //the weights dumped by order "F" from numpy stores col first, then row. so here the index seq of wgt exchanged.
                            w=*((*wgt).wgt_ptr+
                                                 j1-((*ck).stride*j0-pad_ub)+
                                                 (i1-((*ck).stride*i0-pad_lr))*(*ck).kernel_h+
                                                 k1*(*ck).kernel_w*(*ck).kernel_h+
                                k0*(*ck).kernel_w*(*ck).kernel_h*(*ifm).ch);
                            t1+=t2*w;
                            if(i0==15 && j0==3 && k0==0){
                                //printf("%d,%d,%d;%d,%d,%d:t2=%.8f,w=%.8f,t1=%.8f\n",i0,j0,k0,i1,j1,k1,t2,w,t1);
                            }
                        }
                    }
                }
                *((*ofm).fm_ptr+i0+j0*(*ofm).w+k0*(*ofm).w*(*ofm).h)=relu_en?(t1>0?t1:0):t1;
            }
        }
    }
}
void maxpool(Featuremap *ifm,Kernel *pk, Featuremap *ofm){
    printf("Running maxpool..\n");
    int i0,j0,k0;
    int i1,j1;
    double t1,t2;
    for(i0=0;i0<(*ofm).w;i0++){
        for(j0=0;j0<(*ofm).h;j0++){
            for(k0=0;k0<(*ofm).ch;k0++){
                t1=-999;
                for(i1=(*pk).stride*i0;i1<MIN((*ifm).w,(*pk).stride*i0+(*pk).kernel_w);i1++){
                    for(j1=(*pk).stride*j0;j1<MIN((*ifm).h,(*pk).stride*j0+(*pk).kernel_h);j1++){
                        t2=*((*ifm).fm_ptr+j1+i1*(*ifm).h+k0*(*ifm).w*(*ifm).h);
                        t1=t2>t1?t2:t1;
                    }
                }
                *((*ofm).fm_ptr+j0+i0*(*ofm).h+k0*(*ofm).w*(*ofm).h)=t1;
            }
        }
    }
}
void fc(Featuremap *ifm,Weight *wgt,bool relu_en,Featuremap *ofm){
    printf("Running fc..\n");
    int i,j,k;
    double t;
    int ifm_size=(*ifm).w*(*ifm).h*(*ifm).ch;
    for(i=0;i<(*ofm).ch;i++){
       t=*((*wgt).bias_ptr+i);
        for(k=0;k<(*ifm).ch;k++){
           for(j=0;j<(*ifm).w*(*ifm).h;j++){
                     t+=*((*ifm).fm_ptr+j+k*(*ifm).w*(*ifm).h) * (*((*wgt).wgt_ptr+k+j*(*ifm).ch+i*ifm_size)) ;
           }
        }
       *((*ofm).fm_ptr+i)=relu_en?(t>0?t:0):t;
    }
}
void softmax(Featuremap *ifm,Featuremap *ofm){
    printf("Running softmax..\n");
    int i,j,k;
    double sum_exp,t;
    int ifm_size=(*ifm).w*(*ifm).h*(*ifm).ch;
    sum_exp=0;
    for(i=0;i<ifm_size;i++){
        t=*((*ifm).fm_ptr+i);
        sum_exp+=exp(t);
    }
    //printf("sum_exp=%lf\n",sum_exp);
    for(i=0;i<ifm_size;i++){
        t=*((*ifm).fm_ptr+i);
        t=exp(t)/sum_exp;
        sum_exp+=exp(t);
        *((*ofm).fm_ptr+i)=t;
    }
}
void load_wgt(char * filename,Weight *wgt,bool is_bias){
    fstream infile(filename);
    if(!infile){
        printf("Error: Failed to open the file : %s \n",filename);
    }
    double a;
    int i=0;
    int len=is_bias?(*wgt).bias_size/sizeof(double):(*wgt).wgt_size/sizeof(double);
    if(is_bias){
        while(infile>>a && i<len){
            *((*wgt).bias_ptr+i)=a;
            //printf("read bias: %lf\n",*((*wgt).bias_ptr+i));
            i++;
        }
        printf("Read Bias file: %s , \t%d/%d float num loaded.\n",filename,i,len);
    }else{
        while(infile>>a && i<len){
            *((*wgt).wgt_ptr+i)=a;
            //printf("read wgt: %lf\n",*((*wgt).wgt_ptr+i));
            i++;
        }
        printf("Read Wgt  file: %s , \t%d/%d float num loaded.\n",filename,i,len);
    }
    infile.close();
}
void printfm(Featuremap *fm){
    int i,j,k;
    for(k=0;k<(*fm).ch;k++){
         printf("[ch%d]\n",k);
         for(j=0;j<(*fm).h;j++){
            printf("[r%d]:",j);
            for(i=0;i<(*fm).w;i++){
                printf("[%d]%.4lf ",i,*((*fm).fm_ptr+i+j*(*fm).w+k*(*fm).w*(*fm).h));
            }
            printf("\n");
        }
    }
    printf("\n");
}
