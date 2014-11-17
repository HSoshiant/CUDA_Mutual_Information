CUDA_Mutual_Information
=======================

Calculating Mutual_Information by CUDA 

Installation :
  1.Please configure the VS and NVCC paths in the following lines on "setup.m" :
      setenv('VSCOMNTOOLS', 'C:/Program Files (x86)/Microsoft Visual Studio 10.0/Common7/Tools/');
      setenv('MW_NVCC_PATH','C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v6.5/bin')
  2.Execute "setup.m" to compile the files
  
Using Instruction :
	Call calc_MI (data_matrix) and pass the input dataset as its parameter.
	The return value is a vector contains Mutual Informations which can convert to matrix by using "squareform" function.
	
