// Compiled: gcc -std=gnu99 -Ofast [-I/opt/local/include -L/opt/local/lib] -o rbm rbm.c -lm -lopenblas 
// (ignore the warning on the return value of fscanf)

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <cblas.h>
#include <time.h>

#define EPOCHS        100000
#define OUTPUT_EPOCHS 50
#define AVERAGES      1
#define BATCH_SIZE    14    // Datasize, if no batch required
#define VARIANCE      0.1
#define MOMENTUM      0.8   // must be < 1
#define WEIGHTDECAY   0.0
#define LRATE         0.01  // large to get a decreasing log-likelihood
#define CDK           1
#define VISIBLES      9    // Visibles should be less than 30
#define HIDDEN        9

typedef struct batch {
  int size;
  double *block;
} Batch;

#define rnd() ((double)random()/RAND_MAX) // RAND_MAX is 2147483647
#define SIGMOID(X)    (1.0 / (1.0 + exp(-(X))))
#define F(X)   ((rnd() < SIGMOID(X)) ? 1 : 0)

// auxiliary functions
int countlines (FILE *fp) {
    int lines = 0;
    char ch;
    while(!feof(fp)) {
        ch = fgetc(fp);
        if(ch == '\n') {
            lines++;
        }
    }
    return(lines);
}


//------------------------------------------------------------------------
//------------------------------------------------------------------------
// DO NOT FORGET!! Wij is the weight between -> Hi <- and -> Vj <-
//                 Ci are the bias of the VISIBLE neurons
//                 Bi are the bias of the HIDDEN neurons
// Also, we are working with BINARY {0,1} neurons
// (this code is NOT valid for real-valued neurons, or {-1,+1} neurons)
//------------------------------------------------------------------------
//------------------------------------------------------------------------

// Useful globals to compute only once the binary numbers to 2^VISIBLES

double **binary_numbers;

void enumerate_binary_at_index (int index, int *data, int *total_index) {
    if (index == VISIBLES) {
        binary_numbers[*total_index] = (double *)malloc(VISIBLES*sizeof(double));
        for(int i=0; i < VISIBLES; ++i) binary_numbers[*total_index][i] = data[i];
        (*total_index)++;
    } else {
        data[index] = 0;
        enumerate_binary_at_index(index+1, data, total_index);
        data[index] = 1;
        enumerate_binary_at_index(index+1, data, total_index);
    }
}

void initialize_binary_numbers (int total) {
    int data[VISIBLES];
    int index = 0;

    binary_numbers = (double **)malloc(total*sizeof(double *));  // reserve space
    enumerate_binary_at_index(0, data, &index);  // fill the space
}

//------------------------------------------------------------------------

double partition_function (const double *w, const double *b, const double *c ) {
    int total = (1 << VISIBLES);  // 2^VISIBLES
    double partfun = 0.0;
    double *tmp = (double *)malloc(HIDDEN * sizeof(double));

    for (int i=0; i < total; ++i) {
      // compute e^xc where x[i] and c[i] are 0 or 1
      double expxc = exp(cblas_ddot(VISIBLES, binary_numbers[i], 1, c, 1));

      // compute Prod(1+e^(...))
      // but tmp must be restored to b, because cblas_dgemv rewrites its content       
      memcpy(tmp,b,HIDDEN * sizeof(double));  // for (int j=0; j < HIDDEN; ++j) tmp[j] = b[j];
      cblas_dgemv (CblasRowMajor, CblasNoTrans, HIDDEN, VISIBLES, 1.0, w, VISIBLES, binary_numbers[i], 1, 1.0, tmp, 1);
      double prod = 1;
      for (int j=0; j < HIDDEN; ++j) prod *= (1.0 + exp(tmp[j]));
      
      partfun += expxc * prod;
    }
    
    free((double *)tmp);
    return partfun;
}


double energy (const double *x, const double *h, const double *w, const double *b, const double *c ) {
  double *tmp = (double *)calloc(HIDDEN, sizeof(double)); // calloc = malloc but initializes to 0
  cblas_dgemv (CblasRowMajor, CblasNoTrans, HIDDEN, VISIBLES, 1.0, w, VISIBLES, x, 1, 1.0, tmp, 1);
  double energy = cblas_ddot(VISIBLES, x, 1, c, 1) + cblas_ddot(HIDDEN, h, 1, b, 1) + cblas_ddot(HIDDEN, h, 1, tmp, 1);
  free((double *)tmp);
  return -energy;
}


double free_energy (const double *x, const double *w, const double *b, const double *c ) {
  double *tmp = (double *)malloc(HIDDEN * sizeof(double));
  memcpy(tmp,b,HIDDEN * sizeof(double)); // for (int j=0; j < HIDDEN; ++j) tmp[j] = b[j];
  cblas_dgemv (CblasRowMajor, CblasNoTrans, HIDDEN, VISIBLES, 1.0, w, VISIBLES, x, 1, 1.0, tmp, 1);
  double fe = cblas_ddot(VISIBLES, x, 1, c, 1);
  for (int j=0; j < HIDDEN; ++j) fe += log(1.0 + exp(tmp[j]));
  free((double *)tmp);
  return -fe;
}


//------------------------------------------------------------------------


void RBNLearningCDK(double **origindata, int datasize, double *w, double *b, double *c, double *points) {

  double *T, *Tk, *H, *B, *C;
  double *tmph, *tmpv;
  double *pHGibbs, *pHData;

  // initializing weights and biases randomly uniform in [-VARIANCE, VARIANCE]
  for (int i=0; i < HIDDEN * VISIBLES; ++i) w[i] = VARIANCE * (2 * rnd() - 1);
  for (int i=0; i < HIDDEN; ++i)            b[i] = VARIANCE * (2 * rnd() - 1);
  for (int i=0; i < VISIBLES; ++i)          c[i] = VARIANCE * (2 * rnd() - 1);
  
  // Batches will be assigned every epoch after shuffling data
  // Here we initialize the data structures to work with batches
  int     batchsize = BATCH_SIZE;
  int lastblocksize = datasize % batchsize;  
  int     numblocks = (datasize / batchsize) + ((lastblocksize == 0) ? 0 : 1);
  
  Batch *data = (Batch *)malloc(numblocks * sizeof(Batch));
  for (int i=0; i < (numblocks-1); ++i) {
    data[i].size  = batchsize;
    data[i].block = (double *)malloc(VISIBLES * data[i].size * sizeof(double));
  }
  data[numblocks-1].size  = ((lastblocksize == 0) ? batchsize : lastblocksize);
  data[numblocks-1].block = (double *)malloc(VISIBLES * data[numblocks-1].size * sizeof(double));

  // Initializing parameter increment  
  double *Dw0, *Db0, *Dc0; // buffer of zeroes for fast initialization
  Dw0 = (double *)calloc(HIDDEN * VISIBLES, sizeof(double));
  Db0 = (double *)calloc(HIDDEN, sizeof(double));
  Dc0 = (double *)calloc(VISIBLES, sizeof(double));
  double *Dw, *Db, *Dc;   // gradient aproximation
  Dw = (double *)malloc(HIDDEN * VISIBLES * sizeof(double));
  Db = (double *)malloc(HIDDEN * sizeof(double));
  Dc = (double *)malloc(VISIBLES * sizeof(double));

  // Temporal storage that does not depend on the batch size
  double  *tmpb = (double *)malloc(HIDDEN * sizeof(double));
  double  *tmpc = (double *)malloc(VISIBLES * sizeof(double));
  double *tmpw1 = (double *)malloc(HIDDEN * VISIBLES * sizeof(double));
  double *tmpw2 = (double *)malloc(HIDDEN * VISIBLES * sizeof(double));
      
  for (int il=0; il < EPOCHS; ++il) {  // EPOCHS iterations
   
    // shuffle data
    for (int is = datasize - 1; is > 0; is--) {
      int j = (int)((is + 1)*rnd());
      double *tmp = origindata[j];
      origindata[j] = origindata[is];
      origindata[is] = tmp;
    }

    // organizing data in batches, in such a way that is suitable for openblas
    int offset = 0;
    for (int i=0; i < numblocks; ++i) {
      for (int j=0; j < data[i].size; ++j)
	for (int k=0; k < VISIBLES; ++k)
	  data[i].block[k * data[i].size + j] = origindata[offset + j][k];
      offset += data[i].size;
    }

    // initializations weights and biases increments to 0
    memcpy(Dw, Dw0, HIDDEN*VISIBLES*sizeof(double));
    memcpy(Db, Db0, HIDDEN*sizeof(double));
    memcpy(Dc, Dc0, VISIBLES*sizeof(double));

    int currentsize = 0;
    
    for (int ib=0; ib < numblocks; ++ib) { // numblocks iterations, one for every batch

      if (currentsize != data[ib].size) {
	// create B & C matrices with current biases  
	currentsize = data[ib].size;
	B = (double *)malloc(HIDDEN * currentsize * sizeof(double));
	for (int i=0; i < HIDDEN; ++i)
	  for (int j=0; j < currentsize; ++j)
	    B[i*currentsize + j] = b[i];
	C = (double *)malloc(VISIBLES * currentsize * sizeof(double));
	for (int i=0; i < VISIBLES; ++i)
	  for (int j=0; j < currentsize; ++j)
	    C[i*currentsize + j] = c[i];
	// create matrices to compute the aproximation to the gradient
	T  = (double *)malloc(VISIBLES * currentsize * sizeof(double));
	Tk = (double *)malloc(VISIBLES * currentsize * sizeof(double));
	H  = (double *)malloc(HIDDEN   * currentsize * sizeof(double));
	tmph = (double *)malloc(HIDDEN   * currentsize * sizeof(double));
	tmpv = (double *)malloc(VISIBLES * currentsize * sizeof(double));
	pHGibbs = (double *)malloc(HIDDEN   * currentsize * sizeof(double));
	pHData  = (double *)malloc(HIDDEN   * currentsize * sizeof(double));
      }

      // initialize matrices
      memcpy(T, data[ib].block, VISIBLES * currentsize * sizeof(double));

      // Gibbs Sampling
      // 1.- Initialize with observed data 
      memcpy(Tk, T, VISIBLES * currentsize * sizeof(double));

      // 2.- Iterate CDK times
      for (int i=0; i < CDK; ++i) {	  

	// 3.- Get h from x
	memcpy(tmph, B, HIDDEN * currentsize * sizeof(double));
	cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, HIDDEN, currentsize, VISIBLES, 1.0, w, VISIBLES, Tk, currentsize, 1.0, tmph, currentsize);
	for (int i=0; i < HIDDEN * currentsize; ++i) H[i] = F(tmph[i]);

	// 4.- Get x from h
	memcpy(tmpv, C, VISIBLES * currentsize * sizeof(double));
	cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans, VISIBLES, currentsize, HIDDEN, 1.0, w, VISIBLES, H, currentsize, 1.0, tmpv, currentsize);
	for (int i=0; i < VISIBLES * currentsize; ++i) Tk[i] = F(tmpv[i]);
      }
      // End of Gibbs Sampling
     
      // Computing intermediate values that are going to be used
      // more than once: pHData, pHGibbs
      memcpy(pHData,  B, HIDDEN * currentsize * sizeof(double)); 
      cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, HIDDEN, currentsize, VISIBLES, 1.0, w, VISIBLES, T, currentsize, 1.0, pHData, currentsize);
      for (int i=0; i < HIDDEN * currentsize; ++i) pHData[i] = SIGMOID(pHData[i]);

      memcpy(pHGibbs, B, HIDDEN * currentsize * sizeof(double));
      cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, HIDDEN, currentsize, VISIBLES, 1.0, w, VISIBLES, Tk, currentsize, 1.0, pHGibbs, currentsize);
      for (int i=0; i < HIDDEN * currentsize; ++i) pHGibbs[i] = SIGMOID(pHGibbs[i]);

      // updating parameter increments Dw, Db and Dc
      // update hidden biases Db
      for (int i=0; i < HIDDEN; ++i) {
	tmpb[i] = 0;
        for (int j=0; j < currentsize; ++j) 
	  tmpb[i] += pHData[i * currentsize + j] - pHGibbs[i * currentsize + j];
	Db[i] = MOMENTUM * Db[i] + (LRATE/currentsize) * tmpb[i];
      }
      // update visible biases Dc
      for (int i=0; i < VISIBLES; ++i) {
	tmpc[i] = 0;
        for (int j=0; j < currentsize; ++j) 
	  tmpc[i] += T[i * currentsize + j] - Tk[i * currentsize + j];
	Dc[i] = MOMENTUM * Dc[i] + (LRATE/currentsize) * tmpc[i];
      }
      // update weights Dw
      cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, HIDDEN, VISIBLES, currentsize, (LRATE/currentsize), pHData,  currentsize, T,  currentsize, 0, tmpw1, VISIBLES);
      cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, HIDDEN, VISIBLES, currentsize, (LRATE/currentsize), pHGibbs, currentsize, Tk, currentsize, 0, tmpw2, VISIBLES);
      cblas_daxpy(HIDDEN * VISIBLES, -1.0, tmpw2, 1, tmpw1, 1); // tmpw1 := tmpw1 - tmpw2
      cblas_dscal(HIDDEN * VISIBLES, MOMENTUM, Dw, 1);          // Dw := MOMENTUM * Dw
      cblas_daxpy(HIDDEN * VISIBLES, 1.0, tmpw1, 1, Dw, 1);     // Dw := Dw + tmpw1
      
      // if batch size will change in the next iteration, free memory since it will be reallocated
      // also, we should free memory if this is the last iteration
      if ((ib == (numblocks - 1)) || ((ib < (numblocks - 1)) && (currentsize != data[ib+1].size))) {
	free((double *)T);
	free((double *)C);
	free((double *)B);
	free((double *)Tk);
	free((double *)H);
	free((double *)pHGibbs);
	free((double *)pHData);
	free((double *)tmph);
	free((double *)tmpv);
      }
    } // end of for (int ib=0; ib < numblocks; ++ib) 

    // Increment weights and biases
    for (int i=0; i < HIDDEN * VISIBLES; ++i) w[i] += Dw[i];
    for (int i=0; i < HIDDEN; ++i)            b[i] += Db[i];
    for (int i=0; i < VISIBLES; ++i)          c[i] += Dc[i];
    
    // Computing log-likelihood every OUTPUT_EPOCHS steps...
    
    if ((il % OUTPUT_EPOCHS) == 0) {
      double logz = log(partition_function (w, b, c));
      double sum_free_energy = 0;
      for (int i=0; i < datasize; ++i) sum_free_energy += free_energy(origindata[i], w, b, c);
      double log_likelihood = -1 * (datasize * logz + sum_free_energy);
      points[il/OUTPUT_EPOCHS] += log_likelihood;
      //if ((il % 10000) == 0) printf("Epoch: %d -> ll= %5.5lf\n", il, log_likelihood);
      //printf("[DEBUG]-------> %d %5.2lf\n",il/OUTPUT_EPOCHS, log_likelihood);
    }
    
  } // end of for (int il=0; il < EPOCHS; ++il)
  
  // free allocated space
  free((double *)tmpb);
  free((double *)tmpc);
  free((double *)tmpw1);
  free((double *)tmpw2);
  free((double *)Db);
  free((double *)Dc);
  free((double *)Dw);
  free((double *)Db0);
  free((double *)Dc0);
  free((double *)Dw0);
  for (int i=0; i < numblocks; ++i) free((double *)data[i].block);
  free((Batch *)data);
}


//-------------------- main ----------------------------------

int main (int argc, char *argv[]) {
  // argv[1] -> input file
  // argv[2] -> output file 
  
  // ------ Initializations ----------------------

  if (VISIBLES >= 31) {
    fprintf(stderr, "Too many visible neurons (> 31)!\n");
    exit(1);
  }
  int total = (1 << VISIBLES);  // that is 2^VISIBLES, where VISIBLES < 31
  initialize_binary_numbers(total);
  
  // -------- Read input file --------------------
  // ---- as many data as lines in input file ----

  FILE *IN_FIL = fopen(argv[1],"r");
  if (IN_FIL == NULL) {
    fprintf(stderr, "Can't open input file!\n");
    exit(1);
  }
  int datasize = countlines(IN_FIL);
  fclose(IN_FIL);

  double **data = (double **)malloc(datasize*sizeof(double *));
  for (int i=0; i < datasize; ++i)
      data[i] = (double *)malloc(VISIBLES*sizeof(double));

  IN_FIL = fopen(argv[1],"r");
  for (int i=0; i < datasize; ++i)
    for (int j=0; j < VISIBLES; ++j)
      fscanf(IN_FIL, "%lf", &data[i][j]);
  fclose(IN_FIL);

  // --------------------------------------------

  // -------- Learning with CD-K -----------

  double *w, *b, *c;   // parametres to be learned
  w = (double *)malloc(HIDDEN * VISIBLES * sizeof(double));
  c = (double *)malloc(VISIBLES * sizeof(double));
  b = (double *)malloc(HIDDEN * sizeof(double));

  // Compute results...

  int num_points = (int)(EPOCHS / OUTPUT_EPOCHS);
  double *points = (double *)calloc(num_points, sizeof(double));

  for (int i=0; i < AVERAGES; ++i) RBNLearningCDK(data, datasize, w, b, c, points);

  // Output results
  for (int i=0; i < num_points; ++i) printf("%d\t%f\n", i*OUTPUT_EPOCHS, points[i]/AVERAGES);

  // free allocated space
  free((double *)b);
  free((double *)c);
  free((double *)w);
  for (int i=0; i < datasize; ++i) free((double *)data[i]);
  free((double **)data);
  free((double *)points);
  for (int i=0; i < total; ++i) free((double *)binary_numbers[i]);
  free((double **)binary_numbers);
}

