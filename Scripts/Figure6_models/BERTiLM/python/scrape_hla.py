"""
Scrape NetMHCpan data from web.
REQUIRES: out_dir - a string; path to directory in which to save output files
MODIFIES: none
RETURNS:  none
"""

import argparse
import os
import re
import requests

from bs4 import BeautifulSoup

def main():
    parser = argparse.ArgumentParser(prog = 'NetMHCpan Scraper',
                                     description = 'Scrape MHCI ligands from NetMHCpan 4.1.')
    parser.add_argument('-o', '--out_dir', dest = 'out_dir') 
    args = parser.parse_args()

    r  = requests.get('https://services.healthtech.dtu.dk/services/NetMHCpan-4.1/')
    data = r.text
    soup = BeautifulSoup(data)
    
    ## ligands links
    links = soup.find_all('a', href = re.compile('^./suppl/MHCI_ligands/HLA'))
    
    ## download data from all links
    for link in links:
        hla_link = link.get('href')
        file_url = 'https://services.healthtech.dtu.dk/services/NetMHCpan-4.1/' + hla_link[1:]
        req_file = requests.get(file_url)
        req_text = req_file.text
        ## last 10 characters
        hla_name = hla_link[-10:]
        out_file = os.path.join(args.out_dir, hla_name + '.txt')
        ## save MS ligand data
        textfile = open(out_file, 'w+')
        textfile.write(req_text)
        textfile.close()

if __name__ == "__main__":
    main()