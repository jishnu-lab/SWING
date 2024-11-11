# Step One: Cleaning your data and creating your vocabulary files

## Cleaning/Getting Data:

Right now, this package only supports the HLA (sourced from NetMHCpan) and MUTINT datasets, which we have already cleaned/prepared for this project.

The prepared data can be found in csv format here: `/ix/djishnu/Anna/Summer2023/swing_roberta/data/raw/NP_nature_pgen_skempitest_muts_tracker_shuffled.csv`

***scrape_hla.py***:
To scrape the HLA data from the web, run the following command and change the part in brackets as desired:
```
python3 scrape_hla.py --out_dir [path/where/you/want/to/save/the/files]
```
        
The scraped data must then be cleaned (and optionally merged with IEDB data, but this is not something that I have tested yet). The data is cleaned with the build_hla_files.py program, which requires you to already have the data downloaded from IEDB and NetMHCpan, in addition to a chain file that specifies the chain sequences. This chain file can be found at ```/ix/djishnu/Anna/Summer2023/swing_roberta/data/raw/chain-sequence.tsv```.
    
***build_hla_files.py***:
To clean the HLA data, run the following command and change the parts in brackets as desired:
```
python3 build_hla_files.py --iedb_dir [path/to/iedb/data/files] \
                           --netmhc_dir [path/to/netmhc/data/files] \
                           --chain_file [path/to/chain/data/file] \
                           --out_dir [path/where/you/want/to/save/the/files] \
                           [--truncate]
```
        
This will create two different types of files and save them to the provided directory. One will have the suffix iedb_dupremoved.csv and contains the cleaned and merged IEDB and NetMHC data, while the other will have the suffix chains.csv and contains just the cleaned NetMHC data (this is what we have used and are using as of 05 Aug 2023). 
        
Using this program does require you to have cleaned IEDB data, which can be downloaded from [IEDB](https://iedb.org/home_v3.php). At this site, you can download a large .csv file of all HLA MHC Class I sequences, and then clean it using iedb_cleaner.py. In the event that you do not want to go download this data yourself, you can find a previously downloaded version at `/ix/djishnu/Anna/Summer2023/swing_roberta/data/raw/mhc_ligand_full.csv`.
        
***iedb_cleaner.py***:
To clean the IEDB data, run the following command and change the part in brackets as desired:
```
python3 iedb_cleaner.py [path/to/iedb/file] \
                        --out_dir [path/where/you/want/to/save/the/files]
```

## Creating Vocabulary Files:
    
The vocabulary files contain all of the information necessary for tokenization and training the model. The main step here is to k-merize the data, which transforms our two sequences into one sequence of 'words'. To k-merize the data, we use SWING, which slides one sequences over another and translates the corresponding pairs of amino acids into scores that are concatenated. k of these scores are concatenated to create a k-mer, which we can think of as a word. Thus, our long list of numbers becomes something like a sentence, and we can use NLP methods on it. 

This k-merization is performed in build_vocabulary_files.py, which can be run from the command line. It requires the user input `k`, the number of scores/characters to not overlap (`sub_size` - default is 1), and, optionally, the number of positions (`l` - default is 2) to include on either side of this mutation when creating the sliding window. The sliding window will end up being of length $2l+1$, and if `l` is not provided, then the entire second sequence will be used as the sliding window. Finally, a frequency argument `freq` can be provided. If in $(0, 1)$, then it will filter to the top `freq` most frequent k-mers. If `freq` is greater than or equal to $1$, then it will filter to k-mers occurring at least `freq` times.

    
***build_vocabulary_files.py***:
To build the vocabulary files, run the following command and change the parts in brackets as desired:
```
python3 build_vocabulary_files.py --data_dir [path/to/csv/data/files] \
                                  --out_dir [path/where/you/want/to/save/the/files] \
                                  --k [number of sub-units in k-mer] \
                                  --sub_size [number of non-overlaps] \
                                  --l [half size of window] \
                                  --type ['HLA' or 'MUTINT'] \
                                  --freq [filtering frequency]
```
                                          
This will create three different types of files and save them to the provided directory in separate sub-directories. One type is with the suffix `vocab_window_encodings.csv` that contains the window encodings (`window_encodings/`). Another type is with the suffix `vocab_k*_subsize*.txt` that contains only the k-mers (`kmers_txt/`). And the third type is with the suffix `vocab_k*_subsize*.csv` that contains the cleaned/formatted data as well as the k-mers (`kmers_csv/`). Within `kmers_csv/` and `kmers_txt/` there will be two subdirectories: one with the unfiltered and one with the filtered k-mer data. 

        
## Example

```
python3 swing_roberta/build_vocabulary_files.py \
        --data_dir ../npflip_nature_mut_wt_merged_groups.csv \
        --out_dir ./ \
        --k 7 \
        --sub_size 7 \
        --l 1 \
        --freq -1 \
        --type 'MUTINT'
```
