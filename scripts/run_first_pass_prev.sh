for file in config/prev/*.yaml; 
    do python -m treccast.main -c $file -y $1;
done