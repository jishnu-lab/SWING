# SWING
Sliding Window INteraction Grammar 

### Table of Contents
- [Description](#description)
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

[Back To The Top](#read-me-template)

---

## How To Use

#### Installation
#### API Reference

```html
    <p>dummy code</p>
```

#### Dependencies
- pandas (v)
- numpy (v)
- random (v)
- sklearn (v)
- gensim (v)
- pickle (v)
- xgboost (v 1.6.1)
[Back To The Top](#read-me-template)

## Code Documentation
### Language Generation:

#### get_window_encodings(*df*)  
  
Takes a pandas dataframe where each row represents a protein-protein/peptide-protein interaction.  
  
Customization includes setting the interactor protein and the peptide window. In the pMHC context, the epitope defines the peptide window. In the missense mutation pertubation context, the window_k parameter defines the size of the window and the mutation defines the position. Additionally, the scale used to calculate the score can be altered.  
  
The function returns a list of score encodings strings that each represent a PPI. The ends of the encodings include padding from the sliding window process. These encodings will be broken into k-mers for the embedding model. \n
  <dl>
	  <dt> df: a string path to the location of the file </dt>
		  <dd>The file must have a column for the interactor protein sequence, target protein sequence. For the mutation context, the position of the mutation must be provided</dd>
  </dl>
-------------------------------------------------------------------------------------------------------------------------------------------

#### get_kmers_str(*encoding_scores,k=3,shuffle=False, padding_score=9*)  
Takes the encoding scores from get_window_encodings().  
  
Customization includes setting size of the kmers (k), a shuffle option, and the integer defining the padding score.  
  
This function returns a list of lists of overlapping k-mers of specified size k, removing k-mers of only padding. Each list of k-mers are specific to each of the PPIs. This output is compatible with gensims
  
<dl>
	<dt>encoding_scores: a list of lists </dt> 
		<dd>The list contains a list for each PPI. Each PPI list is composed of one string with the encodings </dd>
  <dt>k</dt>: int
    <dd>Defines the size of the k-mers, default=7</dd>
  <dt>shuffle:</dt>
    <dd>Whether the k-mers are shuffled. Shuffling may prevent overfitting based on position of the k-mers. default=False</dd>
  <dt>padding_score: int</dt>
    <dd>Defines the number assigned to the padding. This number should be outside of the range of the scores given to AA pairs.</dd>
</dl>
-------------------------------------------------------------------------------------------------------------------------------------------

#### get_corpus(matrix, tokens_only=False)
Takes in the k-mers created by the get_kmers_str() function.  

Returns a Doc2Vec TaggedDocuments entities for each PPI to be used in a Doc2Vec model.

<dl>
  <dt>matrix: a list of lists</dt>
    <dd> The list that contains a list of k-mers for each PPI</dd>
  <dt>tokens_only:</dt>
    <dd>default=False</dd>
</dl>

-------------------------------------------------------------------------------------------------------------------------------------------

## Author Info
