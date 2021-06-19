#2.	Computing prior causal probabilities via an L2-regularized extension of stratified LD-score regression (S-LDSC).
#--------------------------------------------------------------------------------------------
"""
mkdir -p output

python polyfun.py \
    --compute-h2-L2 \
    --no-partitions \
    --output-prefix output/testrun \
    --sumstats example_data/sumstats.parquet \
    --ref-ld-chr example_data/annotations. \
    --w-ld-chr example_data/weights.


The parameters we provided are the following:

--compute-h2-L2 - this tells PolyFun to compute per-SNP heritabilities via an L2-regularized 
S-LDSC

--no-partitions - this tells PolyFun to not partition SNPs into bins based on their estimated 
per-SNP heritabilities. 
This makes the computations slightly faster. You should only provide this flag if you are only 
interested 
in L2-regularized estimation of per-SNP heritabilities.

--output-prefix output/testrun - this specifies the prefix of all the PolyFun output files.

--sumstats - this specifies an input summary statistics file
 (created via the munge_polyfun_sumstats.py script).

--ref-ld-chr - this is the prefix of the LD-score and annotation files that S-LDSC uses.
 These are similar to the standard S-LDSC input files with an important addition: The annotation files must include 
 columns called A1,A2 for effect and alternative alleles 
 (because unfortunately SNP rsid is not a unique SNP identifier). Additionally, it is strongly 
 recommended that the LD-score files also include columns called A1,A2,
  to prevent excluding multiple SNPs with the same rsid from the estimation stage. PolyFun will 
  accept files with either .gz or .parquet extension (parquet is faster)

--w-ld-chr - this is the prefix of the LD-score weight files. These files should be similar
 to standard LD-score files, but with only one 'annotation' 
  that includes weights. These weights should be equal to the (non-stratified) LD-scores of 
  the SNPs, when computed using plink files that include only the set of SNPs used for fitting S-LDSC. As before, it is strongly recommended that these files include A1,A2 columns.


This will create 2 output files for each chromosome: output/testrun.
<CHR>.snpvar_ridge.gz and output/testrun.<CHR>.snpvar_ridge_constrained.gz. 

For example, here is the output for the top 10 SNPs in chromosome 1:
(seen with cat output/testrun.22.snpvar_ridge_constrained.gz | zcat | head

The column called 'SNPVAR' contains truncated per-SNP heritabilities, 
which can be used directly as prior causal probabilities in fine-mapping
"""

#3.	Computing prior causal probabilities non-parametrically
#----------------------------------------------------------------
#This is done in four stages:
"""
1. Create a munged summary statistics file in a PolyFun-friendly parquet format.
This is done exactly as in step 1 of Approach 2 (see above).

2. Run PolyFun with L2-regularized S-LDSC
mkdir -p output

python polyfun.py \
    --compute-h2-L2 \
    --output-prefix output/testrun \
    --sumstats example_data/sumstats.parquet \
    --ref-ld-chr example_data/annotations. \
    --w-ld-chr example_data/weights.

This is done similarly to step 2 of Approach 2 (see above), except that you should
 remove the --no-partitions flag
3. Compute LD-scores for each SNP bin
python polyfun.py \
    --compute-ldscores \
    --output-prefix output/testrun \
    --bfile-chr example_data/reference. \
    --chr 1

 Here is an example that computes LD-scores for only chromosome 1 by downloading precomputed UK Biobank LD matrices:

mkdir -p LD_cache
python polyfun.py \
    --compute-ldscores \
    --output-prefix output/testrun \
    --ld-ukb \
    --ld-dir LD_cache \
    --chr 1

4. Re-estimate per-SNP heritabilities via S-LDSC
python polyfun.py \
    --compute-h2-bins \
    --output-prefix output/testrun \
    --sumstats example_data/sumstats.parquet \
    --w-ld-chr example_data/weights.

This script will output files with re-estimated per-SNP heritabilities that can be used directly for fine-mapping.
"""

import numpy as np
import pandas as pd
import os
import sys
import logging
from tqdm import tqdm


SNP_COLUMNS = ['CHR', 'SNP', 'BP', 'A1', 'A2']
LONG_RANGE_LD_REGIONS = []
LONG_RANGE_LD_REGIONS.append({'chr':6, 'start':25500000, 'end':33500000})
LONG_RANGE_LD_REGIONS.append({'chr':8, 'start':8000000, 'end':12000000})
LONG_RANGE_LD_REGIONS.append({'chr':11, 'start':46000000, 'end':57000000})
DEFAULT_REGIONS_FILE = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'ukb_regions.tsv.gz')

print(DEFAULT_REGIONS_FILE)

class TqdmUpTo(tqdm):
    """
        taken from: https://github.com/tqdm/tqdm/blob/master/examples/tqdm_wget.py
    """

    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None: self.total = tsize            
        self.update(b * bsize - self.n)



''' Logger class (for compatability with LDSC code)'''
class Logger(object):
    def __init__(self):
        pass
    def log(self, msg):
        logging.info(msg)


class TqdmHandler(logging.StreamHandler):
    def __init__(self):
        logging.StreamHandler.__init__(self)

    def emit(self, record):
        msg = self.format(record)
        tqdm.write(msg)
        

def check_package_versions():
    from pkg_resources import parse_version
    if parse_version(pd.__version__) < parse_version('0.25.0'):
        raise ValueError('your pandas version is too old --- please update pandas')
        
    try:
        import pandas_plink
    except (ImportError, ModuleNotFoundError):
        raise ValueError('\n\nPlease install the python package pandas_plink (using either "pip install pandas-plink" or "conda install -c conda-forge pandas-plink")\n\n')
    
    
def set_snpid_index(df, copy=False, allow_duplicates=False):
    if copy:
        df = df.copy()
    is_indel = (df['A1'].str.len()>1) | (df['A2'].str.len()>1)
    df['A1_first'] = (df['A1'] < df['A2']) | is_indel
    df['A1s'] = df['A2'].copy()
    df.loc[df['A1_first'], 'A1s'] = df.loc[df['A1_first'], 'A1'].copy()
    df['A2s'] = df['A1'].copy()
    df.loc[df['A1_first'], 'A2s'] = df.loc[df['A1_first'], 'A2'].copy()
    df.index = df['CHR'].astype(int).astype(str) + '.' + df['BP'].astype(str) + '.' + df['A1s'] + '.' + df['A2s']
    df.index.name = 'snpid'
    df.drop(columns=['A1_first', 'A1s', 'A2s'], inplace=True)
    
    #check for duplicate SNPs
    if not allow_duplicates:
        is_duplicate_snp = df.index.duplicated()
        if np.any(is_duplicate_snp):
            df_dup_snps = df.loc[is_duplicate_snp]
            df_dup_snps = df_dup_snps.loc[~df_dup_snps.index.duplicated(), ['SNP', 'CHR', 'BP', 'A1', 'A2']]
            error_msg = 'Duplicate SNPs were found in the input data:\n%s'%(df_dup_snps)
            raise ValueError(error_msg)
    return df


def configure_logger(out_prefix):

    logFormatter = logging.Formatter("[%(levelname)s]  %(message)s")
    logger = logging.getLogger()
    logger.setLevel(logging.NOTSET)
    
    consoleHandler = TqdmHandler()
    consoleHandler.setFormatter(logFormatter)
    logger.addHandler(consoleHandler)
    
    fileHandler = logging.FileHandler(out_prefix+'.log')
    fileHandler.setFormatter(logFormatter)
    logger.addHandler(fileHandler)



def get_file_name(args, file_type, chr_num, verify_exists=True, allow_multiple=False):
    if file_type == 'ldscores':
        file_name = args.output_prefix + '.%d.l2.ldscore.parquet'%(chr_num)
    elif file_type == 'snpvar_ridge':
        file_name = args.output_prefix + '.%d.snpvar_ridge.gz'%(chr_num)
    elif file_type == 'taus_ridge':
        file_name = args.output_prefix + '.annot_coeff_ridge.%d.txt'%(chr_num)
    elif file_type == 'taus_nn':
        file_name = args.output_prefix + '.annot_coeff_nn.%d.txt'%(chr_num)
    elif file_type == 'snpvar_ridge_constrained':
        file_name = args.output_prefix + '.%d.snpvar_ridge_constrained.gz'%(chr_num)        
    elif file_type == 'snpvar_constrained':
        file_name = args.output_prefix + '.%d.snpvar_constrained.gz'%(chr_num)        
    elif file_type == 'snpvar':
        file_name = args.output_prefix + '.%d.snpvar.gz'%(chr_num)        
    elif file_type == 'bins':
        file_name = args.output_prefix + '.%d.bins.parquet'%(chr_num)
    elif file_type == 'M':
        file_name = args.output_prefix + '.%d.l2.M'%(chr_num)
        
    elif file_type == 'annot':
        assert verify_exists
        assert allow_multiple
        file_name = []
        for ref_ld_chr in args.ref_ld_chr.split(','):
            file_name_part = ref_ld_chr + '%d.annot.gz'%(chr_num)
            if not os.path.exists(file_name_part):
                file_name_part = ref_ld_chr + '%d.annot.parquet'%(chr_num)
            file_name.append(file_name_part)
        
    elif file_type == 'ref-ld':
        assert verify_exists
        assert allow_multiple
        file_name = []
        for ref_ld_chr in args.ref_ld_chr.split(','):
            file_name_part = ref_ld_chr + '%d.l2.ldscore.gz'%(chr_num)
            if not os.path.exists(file_name_part):
                file_name_part = ref_ld_chr + '%d.l2.ldscore.parquet'%(chr_num)
            file_name.append(file_name_part)
        
    elif file_type == 'w-ld':
        assert verify_exists
        file_name = args.w_ld_chr + '%d.l2.ldscore.gz'%(chr_num)
        if not os.path.exists(file_name):
            file_name = args.w_ld_chr + '%d.l2.ldscore.parquet'%(chr_num)
    
    
    elif file_type == 'bim':
        file_name = args.bfile_chr + '%d.bim'%(chr_num)
    elif file_type == 'fam':
        file_name = args.bfile_chr + '%d.fam'%(chr_num)
    elif file_type == 'bed':
        file_name = args.bfile_chr + '%d.bed'%(chr_num)
    else:
        raise ValueError('unknown file type')
        
    if verify_exists:
        if allow_multiple:
            for fname in file_name:
                if not os.path.exists(fname):
                    raise IOError('%s file not found: %s'%(file_type, fname))
        else:
            if not os.path.exists(file_name):
                raise IOError('%s file not found: %s'%(file_type, file_name))
            
    return file_name
