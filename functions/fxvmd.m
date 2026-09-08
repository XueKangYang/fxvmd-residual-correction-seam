function [ D1 ] = fxvmd(D,flow,fhigh,dt,N,verb,numberIMF)
%FX_EMD: F-X domain empirical mode decomposition along the spatial dimension
%  IN   D:   	intput data 
%       flow:  processing frequency range (lower)
%       fhigh: processing frequency range (higher)
%       dt:    temporal sampling interval
%       N: 	number of IMF to be removed
%       verb:   verbosity flag (default: 0)
%      
%  OUT   D1:  	output data
% 
%  Copyright (C) 2013 The University of Texas at Austin
%  Copyright (C) 2013 Yangkang Chen
%
%  This program is free software: you can redistribute it and/or modify
%  it under the terms of the GNU General Public License as published
%  by the Free Software Foundation, either version 3 of the License, or
%  any later version.
%
%  This program is distributed in the hope that it will be useful,
%  but WITHOUT ANY WARRANTY; without even the implied warranty of
%  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
%  GNU General Public License for more details: http://www.gnu.org/licenses/ 
% 
 
if nargin==0 
 error('Input data must be provided!'); 
end 
 
if nargin==1 
 flow=1; 
 fhigh=124; 
 dt=0.004; 
 N=1; 
 verb=0; 
  
end 
 
[sample,trace ]=size(D);%获取输入数据的采样点数和地震道数 
D1=zeros(sample,trace);%为地震数据配置相同大小的空间 
 
nf=2^nextpow2(sample);%nextpow2函数用来求指数 
nk=2^nextpow2(trace); 
 
% Transform into F-X domain   将数据变换到F-X域 
DATA_FX=fft(D,nf,1);%返回nf个点的傅里叶变换 
DATA_FX0=zeros(nf,trace);%为数据分配相同大小的空间 
 
% First and last samples of the DFT.离散傅里叶变换的第一个和最后一个样本 
ilow  = floor(flow*dt*nf)+1;%将括号中元素取整，值为不大于本身的最大整数 
 
if ilow<1; 
    ilow=1; 
end; 
 
ihigh = floor(fhigh*dt*nf)+1; 
 
if ihigh > floor(nf/2)+1; 
    ihigh=floor(nf/2)+1; 
end 
%在频率域内规定一个范围 
thrsh=0.2; 
 sthresh=4; 
 N1=100; 
 alpha=1; 
% F-X domain ceemd  在频率域内进行CEEMD分解 
for k=ilow:ihigh 
    re=real(DATA_FX(k,:)); 
    im=imag(DATA_FX(k,:)); 
    imfre=vmd(re,'NumIMFs',numberIMF); 
    imfim=vmd(im,'NumIMFs',numberIMF); 
    [mr,nr]=size(imfre); 
    [mi,ni]=size(imfim); 
     
    DATA_FX0(k,:)=re+i*im; 
    if(nr<N || mr<trace) imfre=[imfre;zeros(N-mr,nr)];imfre=[imfre zeros(N,trace-nr)];end 
    if(ni<N || mi<trace) imfim=[imfim;zeros(N-mi,ni)];imfim=[imfim zeros(N,trace-ni)];end     
     if(N==1) 
         DATA_FX0(k,:)=re-(imfre(:,1:N))'+i*im;%频率域内赋值 
    else 
        DATA_FX0(k,:)=re-sum((imfre(:,1:N))')+i*im; 
    end  
    if(mod(k,5)==0 && verb==1) 
        fprintf( 'F %d is done!\n\n',k); 
    end 
end 
 
% Honor symmetries 
for k=nf/2+2:nf 
    DATA_FX0(k,:) = conj(DATA_FX0(nf-k+2,:));%求复数的共轭 
end 
 
% Back to TX (the output) 
D1=real(ifft(DATA_FX0,[],1)); 
D1=D1(1:sample,:); 
 
return 
