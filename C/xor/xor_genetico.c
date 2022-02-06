// Coded by Scratchy
// Algoritmo genetico
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include "genann.h"

/*** RETE NEURALE ***/
#define INPUT 2                 // nodi in ingresso
#define LAYER 1                 // layer nascosti
#define NEURONI 3               // neuroni a layer
#define OUTPUT 1                // nodi in uscita

/*** ALGORITMO GENETICO ***/
#define CAMPIONE 100000
#define SOGLIA 0.80

void checkPtr(void *ptr);

struct struttura{
    double errore;
    long long int epoch;
    genann *ann;
};


int main(){
    long long int i=0, j=0, k=0, genitori=0, epoch=0;
    
    srand(time(0));
    
    // input ed output della rete
    const double input[4][2] = {{0,0},{0,1},{1,0},{1,1}};
    const double output[4] = {0,1,1,0};
    
    // creo la popolazione di reti neurali su cui applicare la selezione
    struct struttura *popolazione = malloc(CAMPIONE * sizeof(struct struttura));
    checkPtr(popolazione);
    
    // inizializzo il campione
    fprintf(stderr,"Inizializzo un campione di %d reti neurali...\n",CAMPIONE);
    for(i=0; i<CAMPIONE; i++){
        popolazione[i].ann = genann_init(INPUT,LAYER,NEURONI,OUTPUT);
        popolazione[i].errore = 0;
        popolazione[i].epoch = 0;
    }
    
    // calcoliamo l'errore di ogni rete con lo scarto quadratico
    fprintf(stderr,"Calcolo l'errore delle singole reti...\n");
    for(i=0; i<CAMPIONE; i++){
        popolazione[i].errore += pow(*genann_run(popolazione[i].ann,input[0]) - output[0],2.0);
        popolazione[i].errore += pow(*genann_run(popolazione[i].ann,input[1]) - output[1],2.0);
        popolazione[i].errore += pow(*genann_run(popolazione[i].ann,input[2]) - output[2],2.0);
        popolazione[i].errore += pow(*genann_run(popolazione[i].ann,input[3]) - output[3],2.0);
    }
    
    // trovo l'errore più piccolo che è stato commesso dalla popolazione
    fprintf(stderr,"Calcolo l'errore minimo...\n");
    double minmax = 1E20;
    for(i=0; i<CAMPIONE; i++){
        if(minmax > popolazione[i].errore){
            minmax = popolazione[i].errore;
        }
    }
    
    // trovo l'errore più grande che è stato commesso dalla popolazione
    fprintf(stderr,"Calcolo l'errore massimo...\n");
    double maxmin = -1E20;
    for(i=0; i<CAMPIONE; i++){
        if(maxmin < popolazione[i].errore){
            maxmin = popolazione[i].errore;
        }
    }
    
    // calcolo un errore accettabile come soglia per stabilire chi sopravvive
    double errSoglia = maxmin * SOGLIA;
    
    // seleziono le reti che hanno commesso un errore <= della soglia calcolata
    for(i=0; i<CAMPIONE; i++){
        if(popolazione[i].errore <= errSoglia){
            popolazione[i].epoch += 1; // questa rete i-esima sopravvive all'epoch
            genitori += 1;
        }
    }
    
    epoch += 1;
    // numero pari per farle accoppiare come dio comanda (lol)
    if(genitori % 2 != 0){
        genitori--;
    }
    
    fprintf(stderr,"EPOCH: %lld\nGenitori: %lld\nErrore minimo: %lf\nErrore massimo: %lf\nSoglia: %lf\n",epoch,genitori,minmax,maxmin,errSoglia);
    
    
    
    // allocazione reti figlie (con pesi temporanei) + reti spazio per reti selezionate
    struct struttura *selezione = malloc(genitori * sizeof(struct struttura));
    struct struttura *figli = malloc((genitori/2) * sizeof(struct struttura));

    // inizializzo le reti figlie
    fprintf(stderr,"\nInizializzo %lld reti neurali figlie...\n",(genitori/2));
    for(i=0; i<(genitori/2); i++){
        figli[i].ann = genann_init(INPUT,LAYER,NEURONI,OUTPUT);
        figli[i].errore = 0;
        figli[i].epoch = epoch;
        
        figli[i].ann->activation_hidden = genann_act_threshold;
        figli[i].ann->activation_output = genann_act_threshold;
    }
    
    // backup reti selezionate
    j = 0;
    for(i=0; i<CAMPIONE; i++){
        if(popolazione[i].epoch == epoch){
            selezione[j].ann = popolazione[i].ann;
            j++;
        }
    }
    
    /*** ACCOPPIAMENTO RETI NEURALI ***/
    // i neuroni di una rete figlia sono la media ponderata tra i neuroni di due reti (genitori);
    // i pesi della media sono gli errori dei genitori
    long long int pesiTotali = selezione[0].ann->total_weights;
    
    k = 0;
    for(i=0; i<(genitori/2); i++){
    
        for(j=0; j<pesiTotali; j++){
            // media pesata
            //figli[k].ann->weight[j] = ((selezione[i].ann->weight[j] * (1/selezione[i].errore)) + (selezione[i+1].ann->weight[j] * (1/selezione[i+1].errore))) / ((1/selezione[i].errore) + (1/selezione[i+1].errore));
            // media semplice
            figli[k].ann->weight[j] = (selezione[i].ann->weight[j] + selezione[i+1].ann->weight[j]) / 2; // media semplice
        }
        
        k++; // scorro le reti figlie
    }
    
    
    // calcoliamo l'errore di ogni rete figlia
    fprintf(stderr,"Calcolo l'errore delle singole reti...\n");
    for(i=0; i<(genitori/2); i++){
        figli[i].errore += pow(*genann_run(figli[i].ann,input[0]) - output[0],2.0);
        figli[i].errore += pow(*genann_run(figli[i].ann,input[1]) - output[1],2.0);
        figli[i].errore += pow(*genann_run(figli[i].ann,input[2]) - output[2],2.0);
        figli[i].errore += pow(*genann_run(figli[i].ann,input[3]) - output[3],2.0);
    }
    
    // trovo l'errore più piccolo
    fprintf(stderr,"Calcolo l'errore minimo delle reti figlie...\n");
    double minmax2 = 1E20;
    for(i=0; i<(genitori/2); i++){
        if(minmax2 > figli[i].errore){
            minmax2 = figli[i].errore;
        }
    }
    
    // trovo l'errore più grande
    fprintf(stderr,"Calcolo l'errore massimo delle reti figlie...\n");
    double maxmin2 = -1E20;
    for(i=0; i<(genitori/2); i++){
        if(maxmin2 < figli[i].errore){
            maxmin2 = figli[i].errore;
        }
    }
    
    fprintf(stderr,"Errore minimo tra le figlie: %lf\nErrore massimo tra le figlie: %lf\n",minmax2,maxmin2);
    
    
    
    /*
    // vediamo cosa predice ora l'ultima dei mohicani
    printf("[%1.f XOR %1.f] = %1.f\n",input[0][0],input[0][1],*genann_run(ann,input[0]));
    printf("[%1.f XOR %1.f] = %1.f\n",input[1][0],input[1][1],*genann_run(ann,input[1]));
    printf("[%1.f XOR %1.f] = %1.f\n",input[2][0],input[2][1],*genann_run(ann,input[2]));
    printf("[%1.f XOR %1.f] = %1.f\n",input[3][0],input[3][1],*genann_run(ann,input[3]));
    
    // salvo i pesi ottimali su file
    FILE *file = fopen("XOR.ann","w");
    checkPtr(file);
    
    genann_write(ann,file);
    
    // libero la memoria e chiudo
    for(i=0; i<CAMPIONE; i++){
        genann_free(popolazione[i].ann);
    }
    
    fclose(file);
    
    */
    
    return 0;
}

void checkPtr(void *ptr){
    
    if(ptr == NULL){
        perror("\nERROR");
        fprintf(stderr,"\n");
        exit(0);
    }
}
