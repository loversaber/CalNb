# CalNb
### The tool for estimating mtDNA bottleneck size in transmission. 

This algorithm of this method is constructed based on the published paper **Transmission of human mtDNA heteroplasmy in the Genome of the Netherlands families: support for a variable-size bottleneck**. 

CalNb was written by Qi Liu in Python and used in the paper **The transmission of human mitochondrial DNA in four-generation pedigrees**(https://doi.org/10.1002/humu.24390)

Please cite these two papers if you use this tool in you paper. **I would appreciate it!**

## Usage
```
time python3 CalNb.py -Nx1 5 -Nx2 6 -nthreads 2 test_input.xls test_output.xls &>log_test &

python3 CalNb.py --help

usage: CalNb.py [-h] [-Nx1 NX1] [-Nx2 NX2] [-nthreads NTHREADS] ipt opt

Based on LMK 2016 Genome Research Paper to Calculate Bottleneck Size(Nb)

positional arguments:
  ipt                 Input File containing the heteroplasmy mutation matrix
  opt                 Output File containing the likelihood value(s) of specific Nx(s)

optional arguments:
  -h, --help          show this help message and exit
  -Nx1 NX1            Range of Nb,Start at Nx1,[1,50] in the Paper.default=1
  -Nx2 NX2            Range of Nb,End at Nx2,[1,50] in the Paper.default=20
  -nthreads NTHREADS  Processing the heteroplasmy dataFrame in parallel(divide into nthreads parts).default=20
```

## Contact
Qi Liu :(lq2021cambridge@gmail.com)
