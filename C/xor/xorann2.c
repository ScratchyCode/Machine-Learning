// Carica una rete neurale già allenata per predire degli output su dei dati
#include <stdio.h>
#include <stdlib.h>
#include "genann.h"

void checkPtr(void *ptr);

int main(){
    const char *nomeFile = "XOR.ann";
    
    FILE *file = fopen(nomeFile,"r");
    checkPtr(file);
    
    // leggo il backup della rete dal file
    genann *ann = genann_read(file);
    checkPtr(ann);

    // dati di input alla rete
    const double input[4][2] = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};

    // osserviamo che predizioni mi da la rete
    printf("[%1.f XOR %1.f] = %1.f\n",input[0][0],input[0][1],*genann_run(ann,input[0]));
    printf("[%1.f XOR %1.f] = %1.f\n",input[1][0],input[1][1],*genann_run(ann,input[1]));
    printf("[%1.f XOR %1.f] = %1.f\n",input[2][0],input[2][1],*genann_run(ann,input[2]));
    printf("[%1.f XOR %1.f] = %1.f\n",input[3][0],input[3][1],*genann_run(ann,input[3]));
    
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
