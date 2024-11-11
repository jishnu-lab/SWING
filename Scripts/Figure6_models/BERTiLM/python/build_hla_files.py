"""
Create combined HLA/MHC datasets for SWING large language model.
Takes data from IEDB and NetMHCpan and combines files for the same HLA sequences. 
REQUIRES: iedb_dir   - a string; path to directory with cleaned IEDB datasets as .csv
          netmhc_dir - a string; path to directory with NetMHCpan datasets as .csv
          chain_file - a string; path to .csv file containing MHC chain sequences
          truncate   - a boolean; whether to truncate to the alpha 1 and alpha 2 sites only
          out_dir    - a string; path to directory in which to save output files
MODIFIES: none
RETURNS:  none
"""

import argparse
import csv    
import glob
import os
import pathlib

import pandas as pd


def main():
    parser = argparse.ArgumentParser(prog = 'HLA MHC File Builder.',
                                     description = 'Merge IEDB and NetMHCpan datasets to create HLA/MHC files for SWING.')
    parser.add_argument('-i', '--iedb_dir')
    parser.add_argument('-n', '--netmhc_dir')
    parser.add_argument('-c', '--chain_file')
    parser.add_argument('-t', '--truncate', action = 'store_true')
    parser.add_argument('-o', '--out_dir')
    args = parser.parse_args()
    
    ## get list of NetMHCpan files
    netmhc_files = list(pathlib.Path(args.netmhc_dir).glob('*.txt'))
    
    ## iterate through NetMHCpan files
    for file in netmhc_files: 
        ## load dataset
        net_mhc = pd.read_csv(file, sep = '\s+', names = ['Epitope', 'Hit', 'MHC'], header = None)
        net_mhc['Set'] = 'MS_ligand'
        
        ## load chain sequences
        chain_seq = pd.read_csv(args.chain_file, sep = '\t').iloc[1:]
        chain_seq = chain_seq[['Label', 'Sequence']]
        
        ## truncate HLA chain sequences to just alpha 1 and 2 domains
        if args.truncate:
            chain_seq['Sequence'] = chain_seq['Sequence'].apply(lambda x: x[25:206])
            epi_length = net_mhc['Epitope'].str.len()
            net_mhc = net_mhc[epi_length <= 10]
        
        ## configure the label to match the chain_seq dataset
        net_mhc['Label'] = net_mhc['MHC'].apply(lambda x: x[:5] + '*' + x[5:] + ' chain')
        
        ## add the MHC sequences to the Epitope data
        mhc_seqs = pd.merge(net_mhc, chain_seq, on = ['Label'], how = 'left')
        
        #read in the IEDB epitope column
        hla_name = str(file).rsplit('/', 1)[-1].rsplit('.txt', 1)[0]
        iedb_file = hla_name[:5] + '*' + hla_name[5:] +'.csv'
        
        ## if there is a file from IEDB, add it to the data frame
        if os.path.isfile(os.path.join(args.iedb_dir, iedb_file)): 
            ## load dataset
            iedb = pd.read_csv('{0}/{1}'.format(args.iedb_dir, iedb_file), usecols = ["Epitope"])
            
            ## remove non standard epitopes
            iedb = iedb[~iedb.Epitope.str.contains('METH')]
            iedb = iedb[~iedb.Epitope.str.contains('OX')]
            iedb = iedb[~iedb.Epitope.str.contains('CYSTL')]
            iedb = iedb[~iedb.Epitope.str.contains('PYRE')]
            
            ## add the MHC sequences to the Epitope data
            iedb['Set'] = 'IEDB'
            iedb['Hit'] = 1
            iedb['MHC'] = hla_name
            iedb['Label'] = iedb['MHC'].apply(lambda x: x[:5] + '*' + x[5:] + ' chain')
            iedb_seqs = pd.merge(iedb, chain_seq, on = ['Label'], how = 'left')
            
            ## combine the two sets
            final_mhc = pd.concat([mhc_seqs, iedb_seqs])
            
            ## remove duplicates
            final_mhc.drop_duplicates(subset = ['Epitope'], keep = 'first', inplace = True, ignore_index = False)
            
            ## make sub-directory 
            try:
                os.makedirs(os.path.join(args.out_dir, 'with_iedb'))
            except FileExistsError:
                pass
            final_mhc.to_csv('{0}/{1}/{2}_iedb_dupremoved.csv'.format(args.out_dir, 'with_iedb', hla_name), index = False)

        ## also save chains
        try:
            os.makedirs(os.path.join(args.out_dir, 'net_mhc'))
        except FileExistsError:
            pass
        
        mhc_seqs.to_csv('{0}/{1}/{2}_chains.csv'.format(args.out_dir, 'net_mhc', hla_name), index = False)

            
if __name__ == "__main__":
    main()