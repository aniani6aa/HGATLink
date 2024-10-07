
## HGATLink: single-cell gene regulatory network inference by fusion of heterogeneous graph attention networks and Transformer
## Usage

1. Install requirements. ```pip install -r requirements.txt```

2. Embedding Train ``` python code/embedding.py```

3. Link prediction ``` python code/main.py```

## Dataset
The dataset is in the dataset folder, which includes the gene embedding matrices that have been learnt under the embeddingpool file, as well as the preprocessed data used for association prediction.
Newly learnt gene embedding information is stored in the embedding information folder.

## Acknowledgement

We are deeply grateful for the valuable code and efforts contributed by the following GitHub repositorie. This contribution has been immensely beneficial to our work.
- GMFGRN (https://github.com/Lishuoyy/GMFGRN)
- GENELink (https://github.com/zpliulab/GENELink)



