# C implmentation of Contrastive Divergence (CD_k)

## Comments

Only tested in Ubuntu & the like (Mint, etc)

Not very useful for large problems, since it computes *explicitly* the partition function.

## Dependencies: openblas, gsl

sudo apt-get install libopenblas-dev

sudo apt-get install libgsl-dev

## Compilation:

gcc -std=gnu99 -Ofast -o rbm rbm.c -lm -lopenblas 
(ignore the warning on the return value of fscanf)

gcc -std=c99 -o bars_and_stripes bars_and_stripes.c -lgsl -lopenblas

gcc -std=c99 -o labeled_shifter_ensemble labeled_shifter_ensemble.c -lgsl -lopenblas

## Not an example of good coding practices

