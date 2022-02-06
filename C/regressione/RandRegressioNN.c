// Ricerca casuale di pesi ottimali data una certa confidenza rispetto al risultato atteso
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include "genann.h"

#define MAXEPOCH 1000           // ricerca massima su una data rete
#define ERRORE 1                // errore massimo per ottenere pesi ottimali
#define ERRMAX 1000
#define MAXTRY 10               // numero di randomizzazioni da raggiungere prima di quittare per mancata convergenza

/*** RETE NEURALE ***/
#define INPUT 2                 // nodi in ingresso
#define LAYER 1                 // layer nascosti
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
    printf("Lettura dati...\n");
    for(i=0; i<lenfile; i++){
        fscanf(file,"%lf %lf %lf\n",&input[i][0],&input[i][1],&output[i]);
        //printf("%lf %lf %lf\n",input[i][0],input[i][1],output[i]); // check sui dati
    }
    
    // #input, #layer, #neuroni, #output
    printf("Creazione della rete neurale...\n");
    genann *ann = genann_init(INPUT,LAYER,NEURONI,OUTPUT);
    checkPtr(ann);
    
    /*
    // settare una funzione di attivazione lineare per i layer nascosti (e non la sigmoid)
    genann_act_linear(ann,0.);
    ann->activation_output = genann_act_threshold;
    */
    
    double err;
    double last_err = ERRMAX;
    unsigned long long int count=0, uscita=0;
    
    printf("Elaborazione...\n");
    
    do{
        count++;
        
        // se non convergo entro un #EPOCH massimo randomizzo la rete e riprovo
        if(count%MAXEPOCH == 0){
            genann_randomize(ann);
            last_err = ERRMAX;
            uscita++;
            printf("\tepoch massimo %lld raggiunto, randomizzo la rete...\n",uscita);
            if(uscita >= MAXTRY){
                printf("Non è stata trovata una convergenza con la statistica richiesta.\nUscita.\n");
                exit(1);
            }
        }
        
        // facciamo un backup della rete
        genann *backup = genann_copy(ann);
        
        // diamo pesi casuali alla nostra rete
        for(i=0; i<ann->total_weights; i++){
            ann->weight[i] += ((double)rand())/RAND_MAX-0.5;
        }
        
        // vediamo cosa abbiamo ottenuto
        // calcoliamo l'errore con lo scarto quadratico tra quello che la rete random predice e quello che dovremmo ottenere
        err = 0;
        for(i=0; i<MAXEPOCH; i++){
            // scorro tutti i dati
            for(j=0; j<lenfile; j++){
                err += pow(*genann_run(ann,input[j]) - output[j],2.0);
            }
        }
        
        // manteniamo i pesi se c'è stato un miglioramento dell'errore dall'iterazione precedente
        // altrimenti riusiamo la rete prima della modifica dei pesi
        if(err < last_err){
            genann_free(backup);
            last_err = err;
        }else{
            genann_free(ann);
            ann = backup;
        }
        
    }while(err > ERRORE);
    
    printf("Finito dopo %lld cicli.\n",count);
    
    // vediamo cosa predice ora la rete
    printf("\nTest della rete:\n");
    printf("[%1.f + %1.f] = %1.f\n",input[0][0],input[0][1],*genann_run(ann,input[0]));
    printf("[%1.f + %1.f] = %1.f\n",input[1][0],input[1][1],*genann_run(ann,input[1]));
    printf("[%1.f + %1.f] = %1.f\n",input[2][0],input[2][1],*genann_run(ann,input[2]));
    printf("[%1.f + %1.f] = %1.f\n",input[3][0],input[3][1],*genann_run(ann,input[3]));
    printf("[%1.f + %1.f] = %1.f\n",input[9][0],input[9][1],*genann_run(ann,input[9]));
    
    // salvo i pesi ottimali su file
    FILE *save = fopen("SOMMATORE.ann","w");
    checkPtr(save);
    
    genann_write(ann,save);
    
    // libero la memoria e chiudo
    genann_free(ann);
    freeMatrix(lenfile,input);
    free(output);
    fclose(file);
    fclose(save);
    
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
