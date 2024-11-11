"""
Extract separate epitope datasets from full IEDB file.
REQUIRES: filename - a string; path to full IEDB .csv file
          out_dir  - a string; path to directory in which to save separate dataset files
MODIFIES: none
RETURNS:  none
"""

import argparse
import csv
import os

from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser(prog = 'IEDB Cleaner',
                                     description = 'Clean epitope datasets from IEDB.')
    parser.add_argument('filename')
    parser.add_argument('-o', '--out_dir', dest = 'out_dir') 
    args = parser.parse_args()
    
    csv_file = open(args.filename, 'r')
    lines = csv_file.readlines()
    
    for line in tqdm(lines):
        ## comma separated file
        q = line.rstrip().split(',')
        
        ## extract desired data from long row of 111 different values
        ## assay_id:  column 0
        ## pmid:      column 3
        ## epi_iri:   column 9
        ## epi_seq:   column 11
        ## source:    column 25
        ## host:      column 43
        ## mhc_restr: column 107
        ## mhc_class: column 110
        desired_cols = [0, 3, 9, 11, 25, 43, 107, 110]
        desired_data = [q[i] for i in desired_cols]
        if desired_data[7] == 'I':
            if '*' in desired_data[6] and ' ' not in desired_data[6]:
                if 'human' in desired_data[5]:
                    out_filename = os.path.join(args.out_dir, desired_data[6] + '.csv')

                    with open(out_filename, 'a+', newline = '') as file:
                        writer = csv.writer(file)
                        fields = ['Epitope', 'MHC']
                        if not os.path.getsize(out_filename) > 0:
                            writer.writerow(fields)
                        writer.writerow([desired_data[3], desired_data[6]])


if __name__ == "__main__":
    main()
    