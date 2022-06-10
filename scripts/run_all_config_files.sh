#!/bin/bash
for file in config/*.yaml; 
    do python -m treccast.main -c $file $@;
done