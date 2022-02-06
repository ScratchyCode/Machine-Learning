// Coded by Pietro Squilla
/*
Classificazione di stati meteorologici associati a cluster di punti in uno spazio TPH;
i cluster possono essere classificati come segue:
    0 = sereno
    1 = nuvolo
    2 = pioggia
    3 = nebbia-sereno
    4 = nebbia-nuvolo
    5 = variabile

Il file di dati ha 4 colonne (temperatura, pressione, umidità, label)
ed N righe (corrispondendi a quante letture sono state fatte)
*/
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include "genann.h"

/*** RETE NEURALE ***/
#define INPUT 3
#define LAYER 10
#define NEURONI 10
#define OUTPUT 7

/*** TRAINING ***/
#define RATE 0.01
#define EPOCH 1000

void checkPtr(void *ptr);
unsigned long long int fileLine(char file[]);
double **createMatrix(long long int rows, long long int columns);
void freeMatrix(long long int rows, double **M);
unsigned long long int randUniforme(unsigned long long int minimo, unsigned long long int massimo);

int main(){
    unsigned long long int i, j, index, line = fileLine("labeled.dat");
    
    // inizializza i pesi della rete all'inizio di ogni init a valori sempre diversi
    srand(time(0));
    
    if(line == 0){
        printf("File vuoto\n");
        exit(1);
    }
    
    double **input = createMatrix(line,3);
    int *output = calloc(line,sizeof(int));
    
    // leggo i dati da file
    fprintf(stderr,"Lettura dati...\n");
    
    FILE *pf = fopen("labeled.dat","r");
    checkPtr(pf);
    
    for(i=0; i<line; i++){
        fscanf(pf,"%lf %lf %lf %d\n",&input[i][0],&input[i][1],&input[i][2],&output[i]);
    }
    
    // creo la rete
    fprintf(stderr,"Creazione rete neurale...\n");
    
    genann *ann = genann_init(INPUT,LAYER,NEURONI,OUTPUT);
    checkPtr(ann);
    
    //ann->activation_hidden = genann_act_sigmoid;
    //ann->activation_output = genann_act_linear;
    //ann->activation_hidden = genann_act_threshold;
    //ann->activation_output = genann_act_threshold;
    
    // addestramento e stima errore
    fprintf(stderr,"Addestramento...\n");
    
    double meteo0[OUTPUT] = {1.,0.,0.,0.,0.,0.,0.};    // sereno
    double meteo1[OUTPUT] = {0.,1.,0.,0.,0.,0.,0.};    // nuvole
    double meteo2[OUTPUT] = {0.,0.,1.,0.,0.,0.,0.};    // coperto
    double meteo3[OUTPUT] = {0.,0.,0.,1.,0.,0.,0.};    // pioggia
    double meteo4[OUTPUT] = {0.,0.,0.,0.,1.,0.,0.};    // grandine
    double meteo5[OUTPUT] = {0.,0.,0.,0.,0.,1.,0.};    // neve
    double meteo6[OUTPUT] = {0.,0.,0.,0.,0.,0.,1.};    // nebbia
    
    double *label;
    const double *tmp;    // vettore di output temporaneo della rete
    double errore;        // errore quadratico medio per valutare l'apprendimento
    
    for(i=0; i<EPOCH; i++){
        
        // training        
        for(j=0; j<line; j++){
            
            // selezione label dell'input
            if(output[j] == 0){
                label = meteo0;
            }else if(output[j] == 1){
                label = meteo1;
            }else if(output[j] == 2){
                label = meteo2;
            }else if(output[j] == 3){
                label = meteo3;
            }else if(output[j] == 4){
                label = meteo4;
            }else if(output[j] == 5){
                label = meteo5;
            }else if(output[j] == 6){
                label = meteo6;
            }else{
                fprintf(stderr,"\nERRORE LABELING\n");
                exit(2);
            }
            
            // discesa stocastica del gradiente se si usa index al posto di j
            index = randUniforme(0,line-1);
            genann_train(ann,input[index],label,RATE);
            //genann_train(ann,input[j],label,RATE);
            
        }
        
        // errore valutato su tutto il dataset per l'epoch i-esimo
        errore = 0;
        for(j=0; j<line; j++){
            tmp = genann_run(ann,input[j]);
            errore += pow(tmp[0] - label[0],2.0);
            errore += pow(tmp[1] - label[1],2.0);
            errore += pow(tmp[2] - label[2],2.0);
            errore += pow(tmp[3] - label[3],2.0);
            errore += pow(tmp[4] - label[4],2.0);
            errore += pow(tmp[5] - label[5],2.0);
            errore += pow(tmp[6] - label[6],2.0);
        }
        
        fprintf(stderr,"\rEpoch: %llu \tErrore2 = %lf",i+1,errore);
    }
    printf("\n\n");
    
    // backup dei pesi della rete su file
    fprintf(stderr,"Salvataggio rete addestrata...\n");
    
    FILE *file = fopen("ATM.ann","w");
    checkPtr(file);
    
    genann_write(ann,file);
    
    // libero la memoria allocata e chiudo il file
    fprintf(stderr,"Fine.\n");
    
    genann_free(ann);
    fclose(file);
    freeMatrix(line,input);
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

unsigned long long int fileLine(char file[]){
    unsigned long long int rows=0;
    char c;
    
    FILE *input = fopen(file,"r");
    if(input == NULL){
        perror("\nError");
        exit(1);
    }
    
    while((c = getc(input)) != EOF){
        if(c == '\n'){
            rows++;
        }
    }
    
    fclose(input);
    
    return rows;
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

void freeMatrix(long long int rows, double **M){

    while(--rows > -1){
        free(M[rows]);
    }
    
    free(M);
    
    return ;
}

// distribuzione uniforme per la discesa stocastica del gradiente
unsigned long long int randUniforme(unsigned long long int minimo, unsigned long long int massimo){
    unsigned long long int intRand = (unsigned long long int)rand();
    unsigned long long int range = massimo - minimo + 1; // +1 rende i valori appartenenti al range compatto
    unsigned long long int x = (intRand % range) + minimo;
    
    return x;
}
