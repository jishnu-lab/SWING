# SWING
Sliding Window INteraction Grammar 

### Table of Contents
- [Description](#description)
- [Input Data](#input-data)
- [How To Use](#how-to-use)
- [Code Documentation](#code-documentation)
- [Author Info](#author-info)

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

For the NCV, only the Epitope, MHC, Hit, and Sequence columns are necessary. For the cross-prediction, the Set should be defined as Train or Test

| Epitope       | MHC         | Set           | Hit           | Sequence              |
| ------------- | ----------- | ------------- | ------------- | --------------------- |       
| AAALIIHHV     | HLA-A02:11  |    Train      |        1      | MAVMAPRTLVLLLSGALAL...|
| AGFAGDDAPR    | HLA-A02:11  |    Test       |        0      | MAVMAPRTLVLLLSGALAL...|


### Missense Mutation Pertubation Context

The Mutated sequence, position of the mutation on the Mutated sequence, the Interactor sequence, and the label (Y2H_score) are necessary
| Mutated Sequence   | Interactor Sequence | Position      | Y2H_score     | 
| ------------------ | ------------------- | ------------- | ------------- |      
| MTMSKEAVTFKDVAVV...| MADEQEIMCKLESIKEI...|    357        |        0      | 
| MWTLVSWVALTAGLVA...| MASPRTRKVLKEVRVQD...|    9          |        1      | 

## How To Use

#### Dependencies
- pandas (v 1.2.4)
- numpy (v 1.20.1)
- sklearn (v 1.3.2)
- gensim (v 4.0.0)
- xgboost (v 1.6.1)
  
### Missense Mutation Pertubation context
A vignette with a step by step explanation of the method has been provided [here](https://github.com/AlisaOmel/SWING/blob/main/Scripts/MutInt_Notebook.ipynb).

### pMHC context
#### Nested Cross Validation (NCV)
To run the nested cross validation  on the class I datasets the following line of code can be used:
```html
    python3 ncv.py --data_set ClassI_training_210.csv --output 'ClassI_NCV_210' --save_embeddings True
    --metric 'polarity' --classifier 'XGBoost' --loops 10 --k 7 --dim 583 --dm 0 --w 11 --min_count 1
    --alpha 0.02349139979145104 --epochs 13 --n_estimators 232 --max_depth 6
    --learning_rate 0.9402316101150048
```
#### Cross Prediction
To run the cross prediction  on the class I datasets the following line of code can be used:
```html
    python3 cross_pred.py --data_set ClassI_crosspred_HLA-A02:02_210.csv --output 'ClassI_HLA-A02:02_210_CI'
    --save_embeddings True --metric 'polarity' --loops 10 --classifier 'XGBoost' --cross_pred_set 'HLA-A02:02'
    --k 7 --dim 583 --dm 0 --w 11 --min_count 1 --alpha 0.02349139979145104 --epochs 13 --n_estimators 232
    --max_depth 6 --learning_rate 0.9402316101150048
```
  

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

#### get_kmers_str(*encoding_scores,k=7,shuffle=False, padding_score=9*)  
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

## Author Info
