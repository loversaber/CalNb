import pandas as pd
import math,argparse,ast
import numpy as np
import numba as nb
from numba import prange,cfunc
from functools import partial
from scipy.integrate import quad
from scipy.special import factorial
from multiprocessing import Pool
import time
import scipy.integrate as si
from numba.types import intc, CPointer, float64
from scipy import LowLevelCallable

import warnings
warnings.filterwarnings('ignore')

@nb.jit
def A1(mf,x,n):
 C=math.lgamma(n+1)-math.lgamma(x+1)-math.lgamma(n-x+1)
 A1_1=math.log(mf)*x
 A1_2=math.log(1-mf)*(n-x)
 item=C+A1_1+A1_2
 A1_result=np.exp(1)**item
 return A1_result

@nb.vectorize(["float64(float64,float64,float64,float64)"])
def formula_6(mt_i,mobs,mn,epsilon):
 item3_A2=np.array([math.exp(1)**(math.log(1-epsilon)*i+math.log(epsilon)*(mt_i-i)+\
           math.log(epsilon)*(mobs-i)+math.log(1-epsilon)*(mn-mt_i-mobs+i)+\
           math.lgamma(mt_i+1)-math.lgamma(mt_i-i+1)-math.lgamma(i+1)+\
           math.lgamma(mn-mt_i+1)-math.lgamma(mobs-i+1)-math.lgamma(mn-mt_i-mobs+i+1)) for i in prange(0,min(mt_i,mobs)+1) if mn-mt_i-mobs+i>=0])
 return item3_A2.sum()

@nb.vectorize(["float64(float64,float64,float64)"])
def A2_numerator_denominator(mf,mt_i,mn):
 item1_A2=A1(mf,mt_i,mn)
 item2_A2=quad(A1,0,1,args=(mt_i,mn))
 return item1_A2/item2_A2[0] 

@nb.jit
def integrate_A(mf,x,n,mobs,mn,epsilon):
 #######A1
 A1_result=A1(mf,x,n)
 #######A2
 mt_i_list=np.arange(mn+1)
 A2_result=np.dot(formula_6(mt_i_list,mobs,mn,epsilon),A2_numerator_denominator(mf,mt_i_list,mn))
 A1_A2=A1_result*A2_result
 return A1_A2

@nb.jit
def do_integrate(func,x,n,mobs,mn,epsilon):
 return quad(func,0,1,args=(x,n,mobs,mn,epsilon))[0]

@nb.jit
def B_B2(ct_i,xf,nf,cn):
 C=math.lgamma(cn+1)-math.lgamma(ct_i+1)-math.lgamma(cn-ct_i+1)
 B1_1=math.log(xf/nf)*ct_i
 B1_2=math.log(1-xf/nf)*(cn-ct_i)
 item=C+B1_1+B1_2
 B2_result=math.exp(1)**item
 return B2_result

@nb.jit
def B_B3(xf,x,n,nf):
 cc2=nf-n+x
 if cc2-xf >= 0:
  C_B3=math.lgamma(nf-n+1)-math.lgamma(xf-x+1)-math.lgamma(nf-n-xf+x+1)
  f8_1=math.lgamma(xf)-math.lgamma(x)
  f8_2=math.lgamma(nf-xf)-math.lgamma(n-x)
  f8_3=math.lgamma(n)-math.lgamma(nf)
  item_B3=C_B3+f8_1+f8_2+f8_3
  B3_result=math.exp(1)**item_B3
 return B3_result

@nb.jit
def B2_B3(ct_i,nf,cn,x,n):
 xf_list=list(range(x,nf))
 result=np.dot(B_B2(ct_i,xf_list,nf,cn),B_B3(xf_list,x,n,nf))
 return result 

@nb.vectorize(["float64(float64,float64,float64,float64,float64)"])
def B2_B3_v(ct_i,nf,cn,x,n):
 result=0
 for xf_list in range(x,nf):
  result+=B_B2(ct_i,xf_list,nf,cn)*B_B3(xf_list,x,n,nf)
 return result

@nb.vectorize(["float64(float64,float64,float64,float64,float64,float64,float64)"])
def formula_B_v(ct_i,cobs,x,n,cn,epsilon,nf):
 return np.dot(formula_6(ct_i,cobs,cn,epsilon),B2_B3_v(ct_i,nf,cn,x,n))#.sum()#np.sum and sum() is not make any sense;because there is just one number

@nb.jit
def formula_1(n,mobs,mn,epsilon,cobs,cn,nf):#xf1 is x_i;xf2 is nf ? nf=1000 epsilon=0.0008
 sigma_n=0
 xf2=nf
 for x in prange(1,n):#simple is (1,n)for formula 8
  A=do_integrate(integrate_A,x,n,mobs,mn,epsilon)
  B=np.sum(formula_B_v(list(range(cn+1)),cobs,x,n,cn,epsilon,nf))
  item_0=A*B
  print(n,x,B,A)
  sigma_n+=item_0
 return sigma_n

def get_mother_child(df4,m,c):
 df_m_c=df4[[m,c]].rename(columns={m:"mother",c:"child"})
 return df_m_c


@nb.jit
def cal_mother_child(row,n,epsilon,nf):
 m=row["mother"]
 c=row["child"]
 mobs,mn=m[1],m[0]
 cobs,cn=c[1],c[0]
 L=formula_1(n,mobs,mn,epsilon,cobs,cn,nf)#formula_1(n,mobs,mn,epsilon,cobs,cn,nf)
 if L==0:
  L_log=-math.inf
 else:
  L_log=math.log(L,1.005)
  L_log10=math.log10(L)
 return L,L_log,L_log10

def parallel_df(df,nthreads,func):
 df_split=np.array_split(df,nthreads)
 pool=Pool(nthreads)
 df_end=pd.concat(pool.map(func,df_split))
 pool.close()
 pool.join()
 return df_end

def cal_df_L(df_split,Nx,epsilon,nf):
 df123=df_split.copy()
 L_syb="L_"+str(Nx)
 L_Log_syb="L_Log_"+str(Nx)
 L_Log_syb2="L_Log10_"+str(Nx)
 df123[L_syb],df123[L_Log_syb],df123[L_Log_syb2]=zip(*df123.apply(lambda row:cal_mother_child(row,Nx,epsilon,nf),axis=1))
 return df123

def End_parallel_cal(df_data,Nx,epsilon,nf,nthreads):
 L_syb="L_"+str(Nx)
 L_Log_syb="L_Log_"+str(Nx)
 L_Log_syb2="L_Log10_"+str(Nx)
 cal_df_L_Parametered=partial(cal_df_L,Nx=Nx,epsilon=epsilon,nf=nf)
 df_data_L=parallel_df(df_data,nthreads,cal_df_L_Parametered)
 multiple_L=1
 for l in df_data_L[L_syb]:
  multiple_L*=l

 print(Nx)
 print(df_data_L)
 Likelihood_value=df_data_L[L_Log_syb].sum()
 Likelihood_value2=df_data_L[L_Log_syb2].sum()
 print("%d\t%.4e\t%.4f\t%.6f"%(Nx,multiple_L,Likelihood_value,Likelihood_value2))
 multiple_L=float("{0:.10e}".format(multiple_L))
 Likelihood_value=float("{0:.10f}".format(Likelihood_value))
 Likelihood_value2=float("{0:.10f}".format(Likelihood_value2))
 return Nx,multiple_L,Likelihood_value,Likelihood_value2


if __name__=="__main__":
 parser=argparse.ArgumentParser(description="Based on LMK 2016 Genome Research Paper to Calculate Bottleneck Size(Nb)")
 #Nx should start from 2 because of formula_8
 parser.add_argument("-Nx1",help="Range of Nb,Start at Nx1,[1,50] in the Paper.default=1",default=1,type=int)
 parser.add_argument("-Nx2",help="Range of Nb,End at Nx2,[1,50] in the Paper.default=20",default=20,type=int)
 parser.add_argument("-nthreads",help="Processing the heteroplasmy dataFrame in parallel(divide into nthreads parts).default=20",default=20,type=int)
 parser.add_argument("ipt",help="Input File containing the heteroplasmy mutation matrix")
 parser.add_argument("opt",help="Output File containing the likelihood value(s) of specific Nx(s)")
 args=parser.parse_args()

 epsilon=0.0006#Sequencing error
 nf=1000#mtDNA population in offspring

 ###Input the mutation matrix
 ###args.ipt needs to contain the following 9 columns(see input example file 'test_input.xls'):
 ###mother,	child,	mobs,	mn,	cobs,	cn,	max_n,	m_maf,	c_maf;
 ###[383, 32],	[386, 40],	32,	383,	40,	386,	386,	0.083550914,	0.103626943;
 ###
 ###mobs the number of copies of the minor allele in the mother, 
 ###𝑚𝑁 the total number of reads in the mother, 
 ###cobs the number of copies of the minor allele in the offspring, 
 ###𝑐𝑁 the total number of reads in the offspring.
 df_data1=pd.read_csv(args.ipt,sep="\t",header=0,converters={"mother": ast.literal_eval,"child": ast.literal_eval})
 df_data=df_data1[(df_data1["m_maf"]>=0.02)&(df_data1["c_maf"]>=0.02)]#filter some mutations according to the MAF threshold in the mother-offspring pairs.

 df_end=pd.DataFrame(columns=("Nx","L","L_Log","L_Log10"))
 for Nx in np.arange(args.Nx1,args.Nx2+1):
  start = time.time()
  nx,mul_L,Likelihood_value,Likelihood_value_log10_sum=End_parallel_cal(df_data,Nx,epsilon,nf,args.nthreads)
  end = time.time()
  print("Elapsed time of %d = %s" % (Nx,end - start))
  df_end=df_end.append({"Nx":nx,"L":mul_L,"L_Log":Likelihood_value,"L_Log10":Likelihood_value_log10_sum},ignore_index=True)
  df_end.to_csv(args.opt,sep="\t",header=True,index=False,mode="w+")
