# SWING
Sliding Window INteraction Grammar 

### Table of Contents
- [Description](#description)
- [Input Data](#input-data)
- [How To Use](#how-to-use)
- [Code Documentation](#code-documentation)
- [Citation](#citation)

---

## Description

An interaction language model for protein-peptide and protein-protein interaction contexts.

#### Contexts

- pMHC
  - Method to predict class I and class II MHC peptide binding.
- Missense mutation pertubations
  - Method to predict whether a protein-protein interaction would occur in the presence of a missense mutation.


---
## Input Data
### pMHC Context

For the SCV, only the Epitope, MHC, Hit, and Sequence columns are necessary. 

For the cross-prediction, the Set should be defined as Train or Test. For prediction of peptides with no known labels, use the nolabel_prediction.py file, the Set should be defiend as Train or Test, and the Hit should be left empty for Test Epitopes.

To cross predict on new peptides/alleles, concatenate the training.csv for the model of your choice found in the Data folder with new data in the format as shown below. For the cross-prediction, the 'Set' column should be defined as 'Train' (from training dataset) or 'Test' (new data). For prediction of peptides with no known labels, use the nolabel_prediction.py file, the 'Set' should be defiend as 'Train' or 'Test', and the 'Hit' column should be left empty for Test Epitopes. 

#### Cross-prediction with known labels: (cross_pred.py)

| Epitope       | MHC         | Set           | Hit           | Sequence              |
| ------------- | ----------- | ------------- | ------------- | --------------------- |       
| AAALIIHHV     | HLA-A02:11  |    Train      |        1      | MAVMAPRTLVLLLSGALAL...|
| AGFAGDDAPR    | HLA-A02:11  |    Test       |        0      | MAVMAPRTLVLLLSGALAL...|

##### Cross-prediction with no  known labels: (nolabel_prediction.py)

| Epitope       | MHC         | Set           | Hit           | Sequence              |
| ------------- | ----------- | ------------- | ------------- | --------------------- |       
| AAALIIHHV     | HLA-A02:11  |    Train      |        1      | MAVMAPRTLVLLLSGALAL...|
| AGFAGDDAPR    | HLA-A02:11  |    Test       |               | MAVMAPRTLVLLLSGALAL...|


### Missense Mutation Pertubation Context

#### General Use (SWING_MutInt_Notebook.ipynb and normal training)
The Mutated sequence (unless WT), position of the mutation on the Mutated sequence (1-indexed, python adjustment in code), the Interactor sequence, and the label (Y2H_score) are the bare minimum necessary to run SWING. We highly recomend you set up your traning data as shown in the [SWING_MutInt_Notebook.ipynb](https://github.com/jishnu-lab/SWING/blob/main/Scripts/SWING_MutInt_Notebook.ipynb).
| Mutated Sequence (unless WT) | Interactor Sequence | Position      | Y2H_score     | 
| ------------------ | ------------------- | ------------- | ------------- |      
| MALDGPEQMELEEGKA...| MTSSYSSSSCPLGCTMA...|    60         |        0      | 
| MARLALSPVPSHWMVA...| MDNKKRLAYAIIQFLHD...|    137        |        1      | 

#### Prediction with no known labels (MutInt_nolabel_prediction.py)
To cross predict on new missense mutations, **concatenate the Mutation_pertubration_model.csv or the data of your choice found in the Data folder with the no label data** in the format as shown below. For the no label prediction, 1. the 'Set' column should be defined as 'Train' (from training dataset) or 'Test' (new, unlabeled data) and the Y2H_score should be left empty for 'Test' mutations 2. columns for the amino acids before and after mutation should be added, and 3. A 'Mutant' or 'WildType' label should be added to the column 'Type'.
Note: Only mutant data should be added for no label prediction (nolabel_pred_set), not wild type. Corresponding type interactions will be added in the background.

| Mutated Sequence (unless WT) | Interactor Sequence | Before_AA | Position | After_AA | Y2H_score | Set | Type |
| ------------------ | ------------------- | ------- | ------- | ------- | ----- | ------ | -------- |
| MALDGPEQMELEEGKA...| MTSSYSSSSCPLGCTMA...|    R    |   60    |  Q      | 0     |  Train | WildType |
| MARLALSPVPSHWMVA...| MDNKKRLAYAIIQFLHD...|    G    |   137   |  S      |       |  Test  |  Mutant  |

## How To Use

#### Dependencies
- pandas (v 1.2.4)
- numpy (v 1.20.1)
- scikit-learn (v 1.3.2)
- gensim (v 4.0.0)
- xgboost (v 1.6.1)
- matplotlib (v 3.3.4)
- python-Levenshtein (v 0.25.1)

### Missense Mutation Pertubation context
#### Standard Cross Validation (SCV)
A vignette with a step by step explanation of the method has been provided [here](https://github.com/jishnu-lab/SWING/blob/main/Scripts/SWING_MutInt_Notebook.ipynb).

#### No Label Prediction
To run no label prediction on mutation data, the following line of code can be used:
```html
python3 MutInt_nolabel_prediction.py --data_set 'data.csv' --output 'no_label_preds' --nolabel_pred_set 'test_set_name' --k 7 --L 1 --metric 'polarity' --padding_score 9 --w 6 --dm 1 --dim 128 --epochs 52 --min_count 1 --alpha 0.08711 --save_embeddings True --n_estimators 375 --max_depth 6 --learning_rate 0.08966
```
### pMHC context
#### Standard Cross Validation (SCV)
To run the standard cross validation  on the Class I datasets the following line of code can be used:
```html
    python3 scv.py --data_set ClassI_training_210.csv --output 'ClassI_SCV_210' --save_embeddings True
    --metric 'polarity' --classifier 'XGBoost' --loops 10 --k 7 --dim 583 --dm 0 --w 11 --min_count 1
    --alpha 0.02349139979145104 --epochs 13 --n_estimators 232 --max_depth 6
    --learning_rate 0.9402316101150048
```
#### Cross Prediction
To run the cross prediction on the Class I datasets the following line of code can be used:
```html
    python3 cross_pred.py --data_set ClassI_crossval_HLA-A02:02_210.csv --output 'ClassI_HLA-A02:02_210'
    --save_embeddings True --metric 'polarity' --loops 10 --classifier 'XGBoost' --cross_pred_set 'HLA-A02:02'
    --k 7 --dim 583 --dm 0 --w 11 --min_count 1 --alpha 0.02349139979145104 --epochs 13 --n_estimators 232
    --max_depth 6 --learning_rate 0.9402316101150048
```
The hyperparameters for the Class II model are:
```html
    --k 7 --dim 146 --dm 0 --w 12 --min_count 1 --alpha 0.03887032752085429 --epochs 13 --n_estimators 341 --max_depth 9 --learning_rate 0.6534638199102993
```
The hyperparameters for the Mixed Class model are:
```html
    --k 7 --dim 74 --dm 0 --w 12 --min_count 1 --alpha 0.03783042872771851 --epochs 10 --n_estimators 269 --max_depth 9 --learning_rate 0.6082359422582875
```
Note ~45G of memory is needed to run the Class I model and ~30G for the Mixed Model.

## Code Documentation
### Language Generation:

#### get_window_encodings(*df, padding_score=9*)  
  
Takes a pandas dataframe where each row represents a protein-protein/peptide-protein interaction.  
  
Customization includes setting the interactor protein and the peptide window. In the pMHC context, the epitope defines the peptide window. In the missense mutation pertubation context, the window_k parameter defines the size of the window and the mutation defines the position. Additionally, the scale used to calculate the score can be altered.  
  
The function returns a list of score encodings strings that each represent a PPI. The ends of the encodings include padding from the sliding window process. These encodings will be broken into k-mers for the embedding model.
  <dl>
	  <dt> df: a string path to the location of the file </dt>
		  <dd>The file must have a column for the interactor protein sequence, target protein sequence. For the mutation context, the position of the mutation must be provided</dd>
	  <dt>padding_score: int</dt>
    		  <dd>Defines the number assigned to the padding. This number should be outside of the range of the scores given to AA pairs. default=9 </dd>
  </dl>
--------------

#### get_kmers_str(*encoding_scores,k=7, padding_score=9*)  
Takes the encoding scores from get_window_encodings().  
  
Customization includes setting size of the kmers (k), a shuffle option, and the integer defining the padding score.  
  
This function returns a list of lists of overlapping k-mers of specified size k, removing k-mers of only padding. Each list of k-mers are specific to each of the PPIs. This output is compatible with gensims
  
<dl>
	<dt>encoding_scores: a list of lists </dt> 
		<dd>The list contains a list for each PPI. Each PPI list is composed of one string with the encodings </dd>
  <dt>k: int</dt>
    <dd>Defines the size of the k-mers, default=7</dd>
  <dt>shuffle:</dt>
    <dd>Whether the k-mers are shuffled. Shuffling may prevent overfitting based on position of the k-mers. default=False</dd>
  <dt>padding_score: int</dt>
    <dd>Defines the number assigned to the padding. This number should be outside of the range of the scores given to AA pairs. default=9 </dd>
</dl>
--------------


#### get_corpus(matrix, tokens_only=False)
Takes in the k-mers created by the get_kmers_str() function.  

Returns a Doc2Vec TaggedDocuments entities for each PPI to be used in a Doc2Vec model.

<dl>
  <dt>matrix: a list of lists</dt>
    <dd> The list that contains a list of k-mers for each PPI</dd>
  <dt>tokens_only:</dt>
    <dd>default=False</dd>
</dl>

--------------

## Citation
Omelchenko, A. A., Siwek, J. C., Chhibbar, P., Arshad, S., Nazarali, I., Nazarali, K., ... & Das, J. (2024). Sliding Window INteraction Grammar (SWING): a generalized interaction language model for peptide and protein interactions. bioRxiv, 2024-05.
 
