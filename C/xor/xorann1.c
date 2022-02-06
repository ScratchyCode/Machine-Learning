// EPOCH di una rete neurale che calcoli lo xor logico tra due input
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "genann.h"

#define EPOCH 1000000

void checkPtr(void *ptr);

int main(){
    // inizializza i pesi della rete all'inizio di ogni init a valori sempre diversi
    srand(time(0));
    
    int i;
    // input di bit di cui valutare lo xor (riassunti in una matrice)
    const double input[4][2] = {{0, 0},{0, 1},{1, 0},{1, 1}};
    // output veri con cui trainare la rete (in input diamo le righe della matrice)
    const double output[4] = {0,1,1,0};
    
    /*
    crea una rete che abbia:
        2 nodi di input;
        1 layer nascosto da 2 neuroni;
        1 nodo di output;
    in ordine prende: input, layer nascosti, # di neuroni dei layer nascosti, output
    */
    
    genann *ann = genann_init(2,1,2,1);
    checkPtr(ann);
    
    // EPOCH sui 4 dati classificati ciclando un numero di volte congruo con il rate di apprendimento
    // la funzione prende in input: la rete da trainare, un array di dati input, un arrai di dati output, il rate di apprendimento
    // il rate di apprendimento deve essere proporzionato a quanto viene trainata la rete
    for(i=0; i<EPOCH; i++){
        genann_train(ann,input[0],output+0,3);
        genann_train(ann,input[1],output+1,3);
        genann_train(ann,input[2],output+2,3);
        genann_train(ann,input[3],output+3,3);
    }
    
    // eseguire la rete osservandone le predizioni
    printf("Lo xor tra [%1.f,%1.f] è %1.f\n",input[0][0],input[0][1],*genann_run(ann,input[0]));
    printf("Lo xor tra [%1.f,%1.f] è %1.f\n",input[1][0],input[1][1],*genann_run(ann,input[1]));
    printf("Lo xor tra [%1.f,%1.f] è %1.f\n",input[2][0],input[2][1],*genann_run(ann,input[2]));
    printf("Lo xor tra [%1.f,%1.f] è %1.f\n",input[3][0],input[3][1],*genann_run(ann,input[3]));
    
    // se sono soddisfatto di come ha appreso posso mantenere il backup di ann su file
    FILE *file = fopen("XOR.ann","w");
    checkPtr(file);
    
    genann_write(ann,file);
    
    // libero la memoria allocata e chiudo il file
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
