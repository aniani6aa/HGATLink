
## HGATLink: single-cell gene regulatory network inference by fusion of heterogeneous graph attention networks and Transformer

## Datasets
The datasets folder contains all the datasets we pre-processed, 14 in total.

## How to use
Step 1:Install the requirements. ``pip install -r requirements.txt``
Step 2: Embedding Train ``python code/embedding.py``
Step 3:Link predictions ``python code/main.py``

## Demo
The Demo folder includes the gene embedding matrices learned under the embeddingspool folder, as well as the four preprocessed data used for link prediction.
1. link prediction (default path to already learned embedding feature file) ``` python Demo/main.py```
2. Re-learned embedding feature file ``` python Demo/embedding.py ``` 
The newly learned gene embedding information is stored in the embeddings folder.

## Acknowledgement

We are deeply grateful for the valuable code and efforts contributed by the following GitHub repositorie. This contribution has been immensely beneficial to our work.
- GMFGRN (https://github.com/Lishuoyy/GMFGRN)
- GENELink (https://github.com/zpliulab/GENELink)



