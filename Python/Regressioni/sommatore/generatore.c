// Coded by Scratchy
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

/*** DATI ***/
#define NUMMAX 100                  // numero di dati analitici da generare
#define TRANSIENTE 20000            // numero di valori casuali da scartare
#define INC 0.1
/*** DATI ***/

/*** FUNZIONE ***/
#define f(x) (sin(x))                 // funzione definita come macro
#define XMIN 0                      // valore del dominio minimo
#define XMAX 2*M_PI               // valore del dominio massimo
#define ERRORE 0.5                // valore dell'incertezza sulle ordinate
/*** FUNZIONE ***/

void checkPtr(void *ptr);
double randReal(double min, double max);

int main(){
    unsigned long long int i, j;
    double x, y;
    
    srand48(time(0));
    
    // transiente (butto via i primi #TRANSIENTE valori pseudocasuali)
    printf("Calcolo il transiente...\n");
    for(i=0; i<TRANSIENTE; i++){
        x = randReal(0,100);
    }
    
    FILE *pf = fopen("dati.dat","w");
    checkPtr(pf);
    
    printf("Genero i dati...\n");
    x = -10;
    y = -10;
    do{
        do{
            fprintf(pf,"%lf %lf %lf\n",x,y,x+y);
            y += INC;
        }while(y <= 10);
        
        y = -10;
        x += INC;
    }while(x <= 10);
    
    fclose(pf);
    
    printf("Fine.\n");
    
    return 0;
}

void checkPtr(void *ptr){
    
    if(ptr == NULL){
        perror("\nERROR");
        fprintf(stderr,"\n");
        exit(0);
    }
}

double randReal(double min, double max){
    double range = (max - min); 
    double denom = RAND_MAX / range;
    
    return min + (lrand48() / denom);
}
