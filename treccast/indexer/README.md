## Resources

To index the TREC CAR dataset we use [trec_car_tools](https://github.com/TREMA-UNH/trec-car-tools). 


## Usage

Resetting the default index and indexing MS MARCO and TREC CAST datasets 
    using default paths.

```
$ python indexing.py --ms_marco --trec_car --reset
```

Indexing only MS MARCO with a custom path to the dataset without resetting
    the index.

```
$ python indexing.py --ms_marco path/to/collection
```