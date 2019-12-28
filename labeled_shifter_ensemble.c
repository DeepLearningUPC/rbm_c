// compile: gcc -std=gnu99 -Ofast -o labeled_shifter_ensemble labeled_shifter_ensemble.c -lm -lgsl -lopenblas

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <gsl/gsl_rng.h>

void generate_labeled_shifter_ensemble (int index, int *storage, int **data, int size, int *total_index) {
    if (index == size) {
        for (int i=0; i < size; ++i)
            data[(*total_index)][i] = data[(*total_index) + 1][i] = data[(*total_index) + 2][i] = storage[i];

        for (int i=0; i < size; ++i)  // to the left if 001
            data[(*total_index)][i + size]   = storage[(i+1)%size];
        data[(*total_index)][2*size] = 0;   data[(*total_index)][2*size + 1] = 0;   data[(*total_index)][2*size + 2] = 1;

        for (int i=0; i < size; ++i)  // unchanged if 010
            data[(*total_index)+1][i + size] = storage[i];
        data[(*total_index)+1][2*size] = 0; data[(*total_index)+1][2*size + 1] = 1; data[(*total_index)+1][2*size + 2] = 0;

        data[(*total_index)+2][size] = storage[size - 1];
        for (int i=1; i < size; ++i)  // to the right if 100
            data[(*total_index)+2][i + size] = storage[i-1];
        data[(*total_index)+2][2*size] = 1; data[(*total_index)+2][2*size + 1] = 0; data[(*total_index)+2][2*size + 2] = 0;

        (*total_index) += 3;
    } else {
        storage[index] = 0;
        generate_labeled_shifter_ensemble (index+1, storage, data, size, total_index);
        storage[index] = 1;
        generate_labeled_shifter_ensemble (index+1, storage, data, size, total_index);
    }
}


int main (int argc, char **argv) {
    // argv[1] --> size, where each data real size is 2 * size + 3

    // initializations...

    // random number generator
    gsl_rng *r = gsl_rng_alloc(gsl_rng_mt19937);

    // dimension data
    int size = atoi(argv[1]);
    int num_patterns = 3 * (1 << size); // 3 * 2^size
    int VISIBLES = 2 * size + 3;

    // space to allocate data
    int **data = (int **)malloc(num_patterns * sizeof(int *));
    for (int i=0; i < num_patterns; ++i) data[i] = (int *)malloc(VISIBLES * sizeof(int));

    // create the bars_and_stripes data...
    int *storage = (int *)malloc(size * sizeof(int *));
    int total_index = 0;

    generate_labeled_shifter_ensemble(0, storage, data, size, &total_index);

    // shuffle before saving
    for (int is = num_patterns - 1; is > 0; is--) {
        int j = gsl_rng_uniform_int (r, is + 1);
        int *tmp = data[j];
        data[j] = data[is];
        data[is] = tmp;
    }


    // "saving", i.e. writing to stdout

    for (int i=0; i < num_patterns; ++i) {
        for (int j=0; j < VISIBLES; ++j)
            printf("%d ", data[i][j]);
        printf("\n");
    }

    /*
    for (int i=0; i < num_patterns; ++i) {
        for (int j=0; j < size; ++j)
            printf("%d ", data[i][j]);

        printf(" -- %d %d %d \n", data[i][2*size], data[i][2*size + 1], data[i][2*size + 2]);

        for (int j=0; j < size; ++j)
            printf("%d ", data[i][j+size]);
        printf("\n\n");
    }
    */



    for (int i=0; i < num_patterns; ++i) free((int *)data[i]);
    free((int **)data);
    gsl_rng_free(r);
}
