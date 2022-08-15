# Encoder

Module containing different types of encoder for generation of embeddings by batch.

## Data generation

Batches are generated with [DataGeneratorMixin](../core/util/data_generator.py). Each document follows this format: `(passage_id, content)`.

## Transformers Encoder

Encoder based on transformers model.  
Only supports [HDF5 file](https://portal.hdfgroup.org/display/HDF5/HDF5) to read/write embeddings. The file stores two datasets: *embeddings* and *passage_ids*.

### Pyserini ANCE encoder

Creates a new embedding file for MS MARCO at default path:
```
$ python -m treccast.encoder.pyserini_ance_encoder --clean --ms_marco
```

Adds embedding for MS MARCO at default path to existing embeddings:
```
$ python -m treccast.encoder.pyserini_ance_encoder --ms_marco
```

More details on arguments:
```
$ python -m treccast.encoder.pyserini_ance_encoder -h
```