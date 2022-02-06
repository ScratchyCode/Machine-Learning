// Ricerca casuale di pesi ottimali data una certa confidenza rispetto al risultato atteso
// Viene usato il metodo della ricottura (in analogia con la tecnica di siderurgia)
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include "genann.h"

#define EPOCH 1000              // ricerca massima su una data rete
#define ERRORE 0.000001         // errore massimo tollerabile sui pesi
#define ERRMAX 1000

/*** RETE NEURALE ***/
#define INPUT 2                 // nodi in ingresso
#define LAYER 1                 // layer nascosti
#define NEURONI 3               // neuroni a layer
#define OUTPUT 1                // nodi in uscita

void checkPtr(void *ptr);

int main(){
    srand(time(0));
    
    // input ed output della rete
    const double input[4][2] = {{0,0},{0,1},{1,0},{1,1}};
    const double output[4] = {0,1,1,0};
    
    // creo la rete neurale
    genann *ann = genann_init(INPUT,LAYER,NEURONI,OUTPUT);
    
    double err;
    double last_err = ERRMAX;
    long long int i, count = 0;
    
    do{
        count++;
        
        // se non convergo in un EPOCH massimo randomizzo la rete e riprovo
        // (ricottura)
        if(count%EPOCH == 0){
            genann_randomize(ann);
            last_err = ERRMAX;
        }
        
        // facciamo un backup della rete
        genann *backup = genann_copy(ann);
        
        // diamo pesi casuali alla nostra rete (raffreddamento)
        for(i=0; i<ann->total_weights; i++){
            ann->weight[i] += ((double)rand())/RAND_MAX-0.5;
        }
        
        // vediamo cosa abbiamo ottenuto
        // calcoliamo l'errore con lo scarto quadratico tra quello che la rete random predice e quello che dovremmo ottenere
        err = 0;
        err += pow(*genann_run(ann,input[0]) - output[0],2.0);
        err += pow(*genann_run(ann,input[1]) - output[1],2.0);
        err += pow(*genann_run(ann,input[2]) - output[2],2.0);
        err += pow(*genann_run(ann,input[3]) - output[3],2.0);
        
        // manteniamo i pesi se c'è stato un miglioramento dell'errore dall'iterazione precedente
        // altrimenti riusiamo la rete prima della modifica dei pesi
        // (selezione raffreddamento ottimale)
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
    printf("[%1.f XOR %1.f] = %1.f\n",input[0][0],input[0][1],*genann_run(ann,input[0]));
    printf("[%1.f XOR %1.f] = %1.f\n",input[1][0],input[1][1],*genann_run(ann,input[1]));
    printf("[%1.f XOR %1.f] = %1.f\n",input[2][0],input[2][1],*genann_run(ann,input[2]));
    printf("[%1.f XOR %1.f] = %1.f\n",input[3][0],input[3][1],*genann_run(ann,input[3]));
    
    // salvo i pesi ottimali su file
    FILE *file = fopen("XOR.ann","w");
    checkPtr(file);
    
    genann_write(ann,file);
    
    // libero la memoria e chiudo
    genann_free(ann);
    fclose(file);
    
    return 0;
}

void checkPtr(void *ptr){
    
    if(ptr == NULL){
        perror("\nERROR");
        fprintf(stderr,"\n");
        exit(0);
    }
}
