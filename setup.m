%clear all
clc
setenv('VSCOMNTOOLS', 'D:/Program Files (x86)/Microsoft Visual Studio 12.0/Common7/Tools/');
setenv('MW_NVCC_PATH','D:/Program Files/NVIDIA GPU/bin')
mex -largeArrayDims calc_MI.cu -v

return

clc
k = 100;
A = rand(k,k);
A = A>0.5;
A = uint8(A);
B =uint16(10);
tic;
C0 = squareform(calc_MI (A,B));
C0=round(C0*100000)/100000;
toc;
clear calc_MI
tic;
%C1 = full(HK_MI1(A));
%C1=round(C1*10000)/10000;
toc;
return;


C1=round(C1*10000)/10000;
C0=round(C0*10000)/10000;
S=sum(sum(C0~=C1))
C0(C0~=C1)
C1(C0~=C1)