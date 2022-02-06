// Training di una rete neurale che calcoli la somma tra due valori positivi
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "genann.h"

/*** TRAINING ***/
#define EPOCH 1000000           // numero di iterazioni del training
#define RATE 0.1               // rate di apprendimento
#define NUMTEST 10             // numero di test sul dataset

/*** RETE NEURALE ***/
#define INPUT 1                 // nodi in ingresso
#define LAYER 10                // layer nascosti
#define NEURONI 3               // neuroni a layer
#define OUTPUT 1                // nodi in uscita

void checkPtr(void *ptr);
long long int dimFile(char file_name[]);
double **createMatrix(long long int rows, long long int columns);
void printMatrix(long long int rows, long long int columns, double **M);
void freeMatrix(long long int rows, double **M);

int main(){
    srand(time(0));
    char *nomeFile = "dati.dat";
    unsigned long long int i, j, lenfile = dimFile(nomeFile);
    
    double *input = calloc(lenfile,sizeof(double));
    double *output = calloc(lenfile,sizeof(double));
    checkPtr(input);
    checkPtr(output);
    
    FILE *file = fopen(nomeFile,"r");
    checkPtr(file);
    
    // leggo i dati
    for(i=0; i<lenfile; i++){
        fscanf(file,"%lf %lf\n",&input[i],&output[i]);
    }
    
    // creo la rete neurale
    genann *ann = genann_init(INPUT,LAYER,NEURONI,OUTPUT);
    checkPtr(ann);
    
    ann->activation_hidden = genann_act_sigmoid;
    ann->activation_output = genann_act_linear;
    
    // training
    unsigned long long int progresso = 0, lavoro = EPOCH*lenfile;
    for(i=0; i<EPOCH; i++){
        for(j=0; j<lenfile; j++){
            genann_train(ann,input,output,RATE);
        }
        fprintf(stderr,"\rTraining -> %.2lf%%",(progresso*100/(double)(lavoro))*100);
        progresso++;
    }
    printf("\n\n");
    
    // eseguire la rete per vedere la bontà dei pesi
    printf("Test su valori del dataset:\n");
    for(i=0; i<NUMTEST; i++){
        printf("f(%lf) = %lf \tresiduo = %lf\n",input[i],*genann_run(ann,input + i),output[i] - *genann_run(ann,input + i));
    }
    
    // se sono soddisfatto di come ha appreso posso mantenere il backup di ann su file
    FILE *backup = fopen("MODELLO.ann","w");
    checkPtr(backup);
    
    genann_write(ann,backup);
    
    // libero la memoria allocata e chiudo il file
    free(input);
    free(output);
    fclose(file);
    fclose(backup);
    genann_free(ann);
    
    return 0;
}

void checkPtr(void *ptr){
    
    if(ptr == NULL){
        perror("\nERROR");
        fprintf(stderr,"\n");
        exit(0);
    }
}

// numero di righe
long long int dimFile(char file_name[]){
    unsigned long long int line = 0;
    char tmp;

	FILE *pf;
	if((pf = fopen(file_name,"r")) == NULL){
		perror("\nError");
		exit(1);
	}
	
	while(!feof(pf)){
	    tmp = fgetc(pf);
	    if(tmp == '\n'){
	        line++;
	    }
	}
	
	fclose(pf);
    
    return line;
}

double **createMatrix(long long int rows, long long int columns){
    long long int i;
    
    double **M = (double **) malloc(rows * sizeof(double *));
    if(M == NULL){
        perror("\nError");
        printf("\n");
        exit(2);
    }
    
    for(i=0; i<rows; i++){
        M[i] = (double *) malloc(columns * sizeof(double));
        if(M == NULL){
            perror("\nError");
            printf("\n");
            exit(3);
        }
    }
    
    return M;
}

void printMatrix(long long int rows, long long int columns, double **M){
    long long int i, j;
    
    printf("\n");
    
    for(i=0; i<rows; i++){
        for(j=0; j<columns; j++){
            printf("%lf ",M[i][j]);
        }
        printf("\n");
    }

    return ;
}

void freeMatrix(long long int rows, double **M){

    while(--rows > -1){
        free(M[rows]);
    }
    
    free(M);
    
    return ;
}
