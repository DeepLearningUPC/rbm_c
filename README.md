# C implementation of Contrastive Divergence (CD_k) 

## Comments

I was trying to compare a C implementation of CD_k with other implementations usually written in Python, Matlab, etc. A *naive* C implementation (not in the repo) was much *slower* than its counterparts. Only by using additional libraries, such as openblas, one is able to remain competitive. Maybe it was obvious but, you know, programming is fun :smiley:

Only tested in Ubuntu & the like (Mint, etc)

Not very useful for large problems, since it computes *explicitly* the partition function. 

## Dependencies: openblas, gsl

`sudo apt-get install libopenblas-dev`

`sudo apt-get install libgsl-dev`

## Compilation and execution:

`gcc -std=gnu99 -Ofast -o rbm_openblas rbm_openblas.c -lm -lopenblas` 
(ignore the warning on the return value of fscanf)

Data goes to `stdout`. Usage: `./rbm <input_data_file> foo`


`gcc -std=gnu99 -Ofast -o bars_and_stripes bars_and_stripes.c -lm -lgsl -lopenblas`
to compile a program that generates data for the bars and stripes problem

Data goes to `stdout`. Usage: `./bars_and_stripes <square linear dimension>`


`gcc -std=gnu99 -Ofast -o labeled_shifter_ensemble labeled_shifter_ensemble.c -lm -lgsl -lopenblas`
to compile a program that generates data for the labeled shifter ensemble problem

Data goes to `stdout`. Usage: `./labeled_shifter_ensemble <N>` where size of each sample will be 2N+3

## Not an example of good coding practices

