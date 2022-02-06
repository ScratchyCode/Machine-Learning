// Training di una rete neurale che calcoli la somma tra due valori positivi
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "genann.h"

/*** TRAINING ***/
#define EPOCH 1000000         // numero di iterazioni del training
#define RATE 0.5              // rate di apprendimento

/*** RETE NEURALE ***/
#define INPUT 2                 // nodi in ingresso
#define LAYER 2                 // layer nascosti
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
    long long int i, j, lenfile = dimFile(nomeFile);
    
    double **input = createMatrix(lenfile,2);
    double *output = calloc(lenfile,sizeof(double));
    checkPtr(input);
    checkPtr(output);
    
    FILE *file = fopen(nomeFile,"r");
    checkPtr(file);
    
    // leggo i dati
    for(i=0; i<lenfile; i++){
        fscanf(file,"%lf %lf %lf\n",&input[i][0],&input[i][1],&output[i]);
        //printf("%lf %lf %lf\n",input[i][0],input[i][1],output[i]); // check sui dati
    }
    
    // #input, #layer, #neuroni, #output
    genann *ann = genann_init(INPUT,LAYER,NEURONI,OUTPUT);
    checkPtr(ann);
    
    printf("Training...\n");
    for(i=0; i<EPOCH; i++){
        for(j=0; j<lenfile; j++){
            genann_train(ann,input[j],(output+j),RATE);
        }
    }
    
    // eseguire la rete per vedere la bontà dei pesi
    printf("Risultati:\n");
    printf("%.1lf \t+ \t%.1lf \t= %.2lf\n",input[0][0],input[0][1],*genann_run(ann,input[0]));
    printf("%.1lf \t+ \t%.1lf \t= %.2lf\n",input[1][0],input[1][1],*genann_run(ann,input[1]));
    printf("%.1lf \t+ \t%.1lf \t= %.2lf\n",input[2][0],input[2][1],*genann_run(ann,input[2]));
    printf("%.1lf \t+ \t%.1lf \t= %.2lf\n",input[10][0],input[10][1],*genann_run(ann,input[10]));
    printf("%.1lf \t+ \t%.1lf \t= %.2lf\n",input[15][0],input[15][1],*genann_run(ann,input[15]));
    printf("%.1lf \t+ \t%.1lf \t= %.2lf\n",input[20][0],input[20][1],*genann_run(ann,input[20]));
    printf("%.1lf \t+ \t%.1lf \t= %.2lf\n",input[30][0],input[30][1],*genann_run(ann,input[30]));
    
    // test
    double addendi[3][3] = {{-1,1},{0.5,0.5},{3,7}};
    printf("\nTest:\n");
    printf("%.1lf \t+ \t%.1lf \t= %.2lf\n",addendi[0][0],addendi[0][1],*genann_run(ann,addendi[0]));
    printf("%.1lf \t+ \t%.1lf \t= %.2lf\n",addendi[1][0],addendi[1][1],*genann_run(ann,addendi[1]));
    printf("%.1lf \t+ \t%.1lf \t= %.2lf\n",addendi[2][0],addendi[2][1],*genann_run(ann,addendi[2]));
    
    // se sono soddisfatto di come ha appreso posso mantenere il backup di ann su file
    FILE *backup = fopen("SOMMATORE.ann","w");
    checkPtr(backup);
    
    genann_write(ann,backup);
    
    // libero la memoria allocata e chiudo il file
    fclose(file);
    fclose(backup);
    genann_free(ann);
    freeMatrix(lenfile,input);
    free(output);
    
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
