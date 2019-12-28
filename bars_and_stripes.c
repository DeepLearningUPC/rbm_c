// compile: gcc -std=gnu99 -Ofast -o bars_and_stripes bars_and_stripes.c -lm -lgsl -lopenblas

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <gsl/gsl_rng.h>

void generate_bars_and_stripes (int index, int *storage, int **data, int size, int *total_index) {
    if (index == size) {
        for (int i=0; i < size; ++i)
            for (int k=0; k < size; ++k)
                data[(*total_index)][i + size * k] = data[(*total_index) + 1][k + size * i] = storage[i];
        (*total_index) += 2;
    } else {
        storage[index] = 0;
        generate_bars_and_stripes(index+1, storage, data, size, total_index);
        storage[index] = 1;
        generate_bars_and_stripes(index+1, storage, data, size, total_index);
    }
}


int main (int argc, char **argv) {
    // argv[1] --> square linear dimension

    // initializations...

    // random number generator
    gsl_rng *r = gsl_rng_alloc(gsl_rng_mt19937);

    // dimension data
    int square_side = atoi(argv[1]);
    int num_patterns = (1 << (square_side + 1)) - 2; // 2^(square_side + 1) - 2
    int VISIBLES = square_side * square_side;

    // space to allocate data
    int **data = (int **)malloc((num_patterns + 2) * sizeof(int *));
    for (int i=0; i < (num_patterns + 2); ++i) data[i] = (int *)malloc(VISIBLES * sizeof(int));

    // create the bars_and_stripes data...
    int *storage = (int *)malloc(square_side * sizeof(int *));
    int total_index = 0;

    generate_bars_and_stripes(0, storage, data, square_side, &total_index);

    // Cleaning data to remove repeated appearances of all 0's and all 1's
    for (int i=1; i < (num_patterns + 2); ++i) data[i-1] = data[i];

    // shuffle before saving
    for (int is = num_patterns - 1; is > 0; is--) {
        int j = gsl_rng_uniform_int (r, is + 1);
        int *tmp = data[j];
        data[j] = data[is];
        data[is] = tmp;
    }


    // "saving", i.e. writing to stdout

    for (int i=0; i < num_patterns; ++i) {
        for (int j=0; j < square_side; ++j)
            for (int k=0; k < square_side; ++k) printf("%d ", data[i][j + square_side * k]);
        printf("\n");
    }


    // "saving", i.e. writing to stdout - square formatted
    /*
      for (int i=0; i < num_patterns; ++i) {
      printf("Pattern num. %d\n", i+1 );
      for (int j=0; j < square_side; ++j) {
      for (int k=0; k < square_side; ++k) printf("%d", data[i][j + square_side * k]);
      printf("\n");
      }
      printf("\n");
      }
    */

    for (int i=0; i < (num_patterns + 1); ++i) free((int *)data[i]);
    free((int **)data);
    gsl_rng_free(r);
}
