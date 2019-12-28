// Compiled: gcc -std=gnu99 -Ofast -o rbm_openblas rbm_openblas.c -lm -lopenblas 
// (ignore the warning on the return value of fscanf)

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <cblas.h>
#include <time.h>

#define EPOCHS        7000
#define OUTPUT_EPOCHS 10
#define AVERAGES      1
#define BATCH_SIZE    30    // Datasize, if no batch required
#define VARIANCE      0.001
#define MOMENTUM      0.0   // must be < 1
#define WEIGHTDECAY   0.0
#define LRATE         0.1   // large to get a decreasing log-likelihood
#define CDK           1
#define VISIBLES      16    // Visibles should be less than 30
#define HIDDEN        16

// RAND_MAX is 2147483647
#define rnd() ((double)random()/RAND_MAX)
#define SIGMOID(X)    (1.0 / (1.0 + exp(-(X))))

// auxiliary functions
int countlines (FILE *fp) {
  int lines = 0;
  char ch;
  while(!feof(fp)) {
    ch = fgetc(fp);
    if(ch == '\n') {
      lines++;
    }
  }  return(lines);
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


void RBNLearningCDK(
                    const double **data,
                    const unsigned long dataSize, // data is a matrix of
                                                  // datasize rows and VISIBLES
                                                  // columns
                    double *w,
                    double *b,
                    double *c,
                    double *points) {

  double *Dw0, *Db0, *Dc0; // buffer of zeroes for fast initialization
  Dw0 = (double *)calloc(HIDDEN * VISIBLES, sizeof(double));
  Db0 = (double *)calloc(HIDDEN, sizeof(double));
  Dc0 = (double *)calloc(VISIBLES, sizeof(double));
  double *Dw, *Db, *Dc;   // gradient aproximation
  Dw = (double *)malloc(HIDDEN * VISIBLES * sizeof(double));
  Db = (double *)malloc(HIDDEN*sizeof(double));
  Dc = (double *)malloc(VISIBLES*sizeof(double));
  // initializations weights and biases increments
  memcpy(Dw, Dw0, HIDDEN*VISIBLES*sizeof(double));
  memcpy(Db, Db0, HIDDEN*sizeof(double));
  memcpy(Dc, Dc0, VISIBLES*sizeof(double));
    
  double *nabla_c_mb, *nabla_b_mb, *nabla_w_mb; // temporal variables
  nabla_w_mb = (double *)malloc(HIDDEN * VISIBLES * sizeof(double));
  nabla_b_mb = (double *)malloc(HIDDEN*sizeof(double));
  nabla_c_mb = (double *)malloc(VISIBLES*sizeof(double));

  double *tmpv, *tmph;
  tmph = (double *)malloc(HIDDEN*sizeof(double));
  tmpv = (double *)malloc(VISIBLES*sizeof(double));
  
  double *hGibbs, *xGibbs;
  hGibbs = (double *)malloc(HIDDEN*sizeof(double));
  xGibbs = (double *)malloc(VISIBLES*sizeof(double));
  
  double *pHGibbs, *pHData;
  pHGibbs = (double *)malloc(HIDDEN*sizeof(double));
  pHData  = (double *)malloc(HIDDEN*sizeof(double));

  // initializing weights and biases randomly uniform in [-VARIANCE, VARIANCE]
  // for (int i=0; i < HIDDEN * VISIBLES; ++i) w[i] = VARIANCE * (2 * rnd() - 1);
  // for (int i=0; i < HIDDEN; ++i)            b[i] = VARIANCE * (2 * rnd() - 1);
  // for (int i=0; i < VISIBLES; ++i)          c[i] = VARIANCE * (2 * rnd() - 1);

  for (int i=0; i < HIDDEN * VISIBLES; ++i) w[i] = VARIANCE * (2 * rnd() - 1);
  for (int i=0; i < HIDDEN; ++i)            b[i] = 0;
  for (int i=0; i < VISIBLES; ++i)          c[i] = 0;
  
  for (int il=0; il < EPOCHS; ++il) {  // EPOCHS iterations

    // shuffle data
    for (int is = dataSize - 1; is > 0; is--) {
      int j = (int)((is + 1)*rnd());
      const double *tmp = data[j];
      data[j] = data[is];
      data[is] = tmp;
    }
    
    int NBATCHES = dataSize / BATCH_SIZE;
    for (int ib=0; ib < NBATCHES; ++ib) {  // NBATCHES times process mini-batch

      // process every data in the mini-batch
      int limitBatch = (ib + 1)*BATCH_SIZE;
      double REAL_SIZE = (limitBatch > dataSize) ? (dataSize - ib * BATCH_SIZE) : BATCH_SIZE;
      
      memcpy(nabla_w_mb, Dw0, HIDDEN*VISIBLES*sizeof(double));
      memcpy(nabla_b_mb, Db0, HIDDEN*sizeof(double));
      memcpy(nabla_c_mb, Dc0, VISIBLES*sizeof(double));
    
      for (int id = ib * BATCH_SIZE; (id < limitBatch) && (id < dataSize); ++id ) {
        
        // Computing intermediate values that are going to be used
        // more than once: pHData
        memcpy(pHData, b, HIDDEN * sizeof(double)); // for (int j=0; j < HIDDEN; ++j)  pHData[j] = b[j];
        cblas_dgemv (CblasRowMajor, CblasNoTrans, HIDDEN, VISIBLES, 1.0, w, VISIBLES, data[id], 1, 1.0, pHData,  1);
        for (int j=0; j < HIDDEN; ++j)
          pHData[j]  = SIGMOID(pHData[j]);
        
        // Gibbs Sampling
        // 1.- Initialize with observed data 
        memcpy(xGibbs, data[id], VISIBLES * sizeof(double)); // for (int i=0; i < VISIBLES; ++i) xGibbs[i] = data[id][i];
        memcpy(tmph, pHData, HIDDEN * sizeof(double)); 
        // 2.- Iterate CDK-1 times
        for (int i=0; i < CDK-1; ++i) {
          for (int j=0; j < HIDDEN; ++j) hGibbs[j] = (rnd() < tmph[j]) ? 1 : 0;	  
          
          memcpy(tmpv,c,VISIBLES * sizeof(double)); // for (int i=0; i < VISIBLES; ++i) tmpv[i] = c[i];
          cblas_dgemv (CblasRowMajor, CblasTrans,   HIDDEN, VISIBLES, 1.0, w, VISIBLES, hGibbs, 1, 1.0, tmpv, 1);
          for (int j=0; j < VISIBLES; ++j) xGibbs[j] = (rnd() < SIGMOID(tmpv[j])) ? 1 : 0;	  

          memcpy(tmph,b,HIDDEN * sizeof(double)); // for (int i=0; i < HIDDEN; ++i) tmph[i] = b[i];
          cblas_dgemv (CblasRowMajor, CblasNoTrans, HIDDEN, VISIBLES, 1.0, w, VISIBLES, xGibbs, 1, 1.0, tmph, 1);
          for (int j=0; j < HIDDEN; ++j) tmph[j] = SIGMOID(tmph[j]);
        }
        // End of Gibbs Sampling
	
        // Last iteration: No sampling, but probabilities.
        for (int j=0; j < HIDDEN; ++j) hGibbs[j] = (rnd() < tmph[j]) ? 1 : 0;	  
        memcpy(tmpv,c,VISIBLES * sizeof(double)); // for (int i=0; i < VISIBLES; ++i) tmpv[i] = c[i];
        cblas_dgemv (CblasRowMajor, CblasTrans,   HIDDEN, VISIBLES, 1.0, w, VISIBLES, hGibbs, 1, 1.0, tmpv, 1); 
        for (int j=0; j < VISIBLES; ++j) xGibbs[j] = SIGMOID(tmpv[j]);	  
                        
        // Computing intermediate values that are going to be used
        // more than once: pHGibbs
        memcpy(pHGibbs,b, HIDDEN * sizeof(double)); // for (int j=0; j < HIDDEN; ++j) pHGibbs[j] = b[j];
        cblas_dgemv (CblasRowMajor, CblasNoTrans, HIDDEN, VISIBLES, 1.0, w, VISIBLES, xGibbs,   1, 1.0, pHGibbs, 1);
        for (int j=0; j < HIDDEN; ++j)
          pHGibbs[j] = SIGMOID(pHGibbs[j]);

        for (int i=0; i < VISIBLES; ++i)   nabla_c_mb[i] += data[id][i] - xGibbs[i];
        for (int i=0; i < HIDDEN; ++i)     nabla_b_mb[i] += pHData[i] - pHGibbs[i];
        for (int i=0; i < HIDDEN; ++i)
          for (int j=0; j < VISIBLES; ++j) nabla_w_mb[i*VISIBLES + j] += (pHData[i] * data[id][j] - pHGibbs[i] * xGibbs[j]);
        
      } // for (int id = ib * BATCH_SIZE; (id < limitBatch) && (id < dataSize); ++id )
      
      for (int i=0; i < HIDDEN * VISIBLES; ++i) nabla_w_mb[i] /= REAL_SIZE;
      for (int i=0; i < HIDDEN;   ++i)          nabla_b_mb[i] /= REAL_SIZE;
      for (int i=0; i < VISIBLES; ++i)          nabla_c_mb[i] /= REAL_SIZE;

      // Increment Dw[][], Db[] and Dc[] - Learning Rule
      for (int i=0; i < HIDDEN * VISIBLES; ++i) Dw[i] = MOMENTUM * Dw[i] + LRATE * (nabla_w_mb[i] - (WEIGHTDECAY/dataSize) * w[i]);
      for (int i=0; i < HIDDEN;   ++i)          Db[i] = MOMENTUM * Db[i] + LRATE * (nabla_b_mb[i] - (WEIGHTDECAY/dataSize) * b[i]);
      for (int i=0; i < VISIBLES; ++i)          Dc[i] = MOMENTUM * Dc[i] + LRATE * (nabla_c_mb[i] - (WEIGHTDECAY/dataSize) * c[i]); 

      // Increment weights and biases
      for (int i=0; i < HIDDEN * VISIBLES; ++i) w[i] += Dw[i];
      for (int i=0; i < HIDDEN; ++i)            b[i] += Db[i];
      for (int i=0; i < VISIBLES; ++i)          c[i] += Dc[i];
            
    } // for (int ib=0; ib < NBATCHES; ++ib)
    
    
    // Computing log-likelihood every OUTPUT_EPOCHS steps...

    if ((il % OUTPUT_EPOCHS) == 0) {
      double Z = partition_function (w, b, c);
      double logz = log(Z);
      double sum_free_energy = 0;
      double sum_probs = 0;
      for (int i=0; i < dataSize; ++i) {
        double tmpFE = free_energy(data[i], w, b, c);
        sum_free_energy += tmpFE;
        sum_probs       += exp(-tmpFE);
      }
      sum_probs /= Z;
      double log_likelihood = -1 * (dataSize * logz + sum_free_energy);
      points[il/OUTPUT_EPOCHS] += log_likelihood;
      // if ((il % 10000) == 0) printf("Epoch: %d -> ll= %5.5lf\n", il, log_likelihood);
      // printf("[DEBUG]-------> %d %5.2lf\n",il/OUTPUT_EPOCHS, log_likelihood);
      printf("%d\t%f\t%f\n", il, log_likelihood, sum_probs);
    }
    
  } // for (int il=0; il < EPOCHS; ++il)
  
  // free allocated space
  free((double *)tmph);
  free((double *)tmpv);
  free((double *)Db);
  free((double *)Dc);
  free((double *)Dw);
  free((double *)Db0);
  free((double *)Dc0);
  free((double *)Dw0);
  free((double *)pHGibbs);
  free((double *)pHData);
  free((double *)hGibbs);
  free((double *)xGibbs);
  free((double *)nabla_w_mb);
  free((double *)nabla_b_mb);
  free((double *)nabla_c_mb);
}


//-------------------- main ----------------------------------

int main (int argc, char *argv[]) {
  /* argv[1] -> input file
     argv[2] -> output file */
  int datasize;
  double **data;
  int total = (1 << VISIBLES);  // that is 2^VISIBLES, where VISIBLES < 31

  int num_points = (int)(EPOCHS / OUTPUT_EPOCHS);
  double *points = (double *)calloc(num_points, sizeof(double));

  // ------ Initializations ----------------------

  initialize_binary_numbers(total);

  // -------- Read input file --------------------
  // ---- as many data as lines in input file ----

  FILE *IN_FIL = fopen(argv[1],"r");
  if (IN_FIL == NULL) {
    fprintf(stderr, "Can't open input file!\n");
    exit(1);
  }
  datasize = countlines(IN_FIL);
  fclose(IN_FIL);

  data = (double **)malloc(datasize*sizeof(double *));
  for (int i=0; i < datasize; ++i)
    data[i] = (double *)malloc(VISIBLES*sizeof(double));

  IN_FIL = fopen(argv[1],"r");
  for (int i=0; i < datasize; ++i)
    for (int j=0; j < VISIBLES; ++j)
      fscanf(IN_FIL, "%lf", &data[i][j]);
  fclose(IN_FIL);

  // ---------------------------------------

  // -------- Learning with CD-K -----------

  double *w, *b, *c;   // parametres to be learned
  w = (double *)malloc(HIDDEN * VISIBLES * sizeof(double));
  c = (double *)malloc(VISIBLES * sizeof(double));
  b = (double *)malloc(HIDDEN * sizeof(double));

  // Compute results...
  for (int i=0; i < AVERAGES; ++i) RBNLearningCDK((const double **)data, datasize, w, b, c, points);

  // Output results
  //for (int i=0; i < num_points; ++i) printf("%d\t%f\n", i*OUTPUT_EPOCHS, points[i]/AVERAGES);

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

