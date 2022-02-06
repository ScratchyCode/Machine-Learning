// Carica una rete neurale già allenata per predire degli output su dei dati
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "genann.h"

void checkPtr(void *ptr);

int main(){
    int i;
    double min1, min2, min3, max1, max2, max3;
    char *meteo[7] = {"sereno","nuvole","coperto","pioggia","grandin","neve","nebbia"};
    
    // leggo i minimi ed i massimi per normalizzare i dati in input
    fprintf(stderr,"Lettura dati per normalizzare gli input...\n");
    FILE *minmax = fopen("minmax.dat","r");
    checkPtr(minmax);
    fscanf(minmax,"MINIMI: %lf %lf %lf\n",&min1,&min2,&min3);
    fscanf(minmax,"MASSIMI: %lf %lf %lf\n",&max1,&max2,&max3);
    fclose(minmax);
    
    // leggo il backup della rete dal file
    const char *nomeFile = "ATM.ann";
    
    FILE *file = fopen(nomeFile,"r");
    checkPtr(file);
    
    fprintf(stderr,"Lettura rete neurale...\n");
    
    genann *ann = genann_read(file);
    checkPtr(ann);
    fclose(file);
    
    double T, P, H;
    printf("Inserisci la temperatura (°C): ");
    scanf("%lf",&T);
    
    printf("Inserisci la pressione (hPa): ");
    scanf("%lf",&P);
    
    printf("Inserisci l'umidità relativa (%%): ");
    scanf("%lf",&H);
    
    // dati di input alla rete (normalizzati come i dati di training)
    double input[3];
    input[0] = (T - min1) / (double)(max1 - min1);
    input[1] = (P - min2) / (double)(max2 - min2);
    input[2] = (H - min3) / (double)(max3 - min3);
    
    const double *output = genann_run(ann,input);
    
    for(i=0; i<7; i++){
        printf("%s\t->\t%lf\n",meteo[i],output[i]);
    }
    
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
