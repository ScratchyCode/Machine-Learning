#include <stdio.h>
#include <stdlib.h>

void checkPtr(void *ptr);
unsigned long long int fileLine(char file[]);
double *max(char *file);
double *min(char *file);

int main(){
    unsigned long long int i, line;
    double T, P, H;
    double tmpT, tmpP, tmpH;
    
    FILE *sereno = fopen("sereno.dat","r");
    FILE *nuvole = fopen("nuvole.dat","r");
    FILE *coperto = fopen("coperto.dat","r");
    FILE *pioggia = fopen("pioggia.dat","r");
    checkPtr(sereno);
    checkPtr(nuvole);
    checkPtr(coperto);
    checkPtr(pioggia);
    
    FILE *dati = fopen("labeled.dat","a+");
    checkPtr(dati);
    
    // labeling + normalizzazione
    line = fileLine("sereno.dat");
    double *minimi = min("merge.dat");     // minimi di tutto il dataset
    double *massimi = max("merge.dat");    // massimi di tutto il dataset
    
    for(i=0; i<line; i++){
        fscanf(sereno,"%lf %lf %lf\n",&T,&P,&H);
        
        if(H >= 100.){
            H = 100.;
        }else if(H < 0.){
            H = 0.;
        }
        
        tmpT = (T - minimi[0]) / (double)(massimi[0] - minimi[0]);
        tmpP = (P - minimi[1]) / (double)(massimi[1] - minimi[1]);
        tmpH = (H - minimi[2]) / (double)(massimi[2] - minimi[2]);
        
        // dati normalizzati
        fprintf(dati,"%lf %lf %lf %d\n",tmpT,tmpP,tmpH,0);
    }
    
    
    line = fileLine("nuvole.dat");
    
    for(i=0; i<line; i++){
        fscanf(nuvole,"%lf %lf %lf\n",&T,&P,&H);
        
        if(H >= 100.){
            H = 100.;
        }else if(H < 0.){
            H = 0.;
        }
        
        tmpT = (T - minimi[0]) / (double)(massimi[0] - minimi[0]);
        tmpP = (P - minimi[1]) / (double)(massimi[1] - minimi[1]);
        tmpH = (H - minimi[2]) / (double)(massimi[2] - minimi[2]);
        
        // dati normalizzati
        fprintf(dati,"%lf %lf %lf %d\n",tmpT,tmpP,tmpH,1);
    }
    
    
    
    line = fileLine("coperto.dat");
    
    for(i=0; i<line; i++){
        fscanf(coperto,"%lf %lf %lf\n",&T,&P,&H);
        
        if(H >= 100.){
            H = 100.;
        }else if(H < 0.){
            H = 0.;
        }
        
        tmpT = (T - minimi[0]) / (double)(massimi[0] - minimi[0]);
        tmpP = (P - minimi[1]) / (double)(massimi[1] - minimi[1]);
        tmpH = (H - minimi[2]) / (double)(massimi[2] - minimi[2]);
        
        // dati normalizzati
        fprintf(dati,"%lf %lf %lf %d\n",tmpT,tmpP,tmpH,2);
    }
    
    
    line = fileLine("pioggia.dat");
    
    for(i=0; i<line; i++){
        fscanf(pioggia,"%lf %lf %lf\n",&T,&P,&H);
        
        if(H >= 100.){
            H = 100.;
        }else if(H < 0.){
            H = 0.;
        }
        
        tmpT = (T - minimi[0]) / (double)(massimi[0] - minimi[0]);
        tmpP = (P - minimi[1]) / (double)(massimi[1] - minimi[1]);
        tmpH = (H - minimi[2]) / (double)(massimi[2] - minimi[2]);
        
        // dati normalizzati
        fprintf(dati,"%lf %lf %lf %d\n",tmpT,tmpP,tmpH,3);
    }
    
    
    fclose(pioggia);
    fclose(coperto);
    fclose(nuvole);
    fclose(dati);
    
    free(minimi);
    free(massimi);
    
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

double *max(char *file){
    long long int i, dim = fileLine(file);
    double max1=-1E20;
    double max2=-1E20;
    double max3=-1E20;
    double *massimi = calloc(3,sizeof(double));
    
    FILE *pf = fopen(file,"r");
    
    for(i=0; i<dim; i++){
        fscanf(pf,"%lf %lf %lf\n",massimi,massimi+1,massimi+2);
        
        if(max1 < massimi[0]){
            max1 = massimi[0];
        }
        
        if(max2 < massimi[1]){
            max2 = massimi[1];
        }
        
        if(max3 < massimi[2]){
            max3 = massimi[2];
        }
    }
    
    massimi[0] = max1;
    massimi[1] = max2;
    massimi[2] = max3;
    
    FILE *minmax = fopen("minmax.dat","a+");
    checkPtr(minmax);
    fprintf(minmax,"MASSIMI: %lf %lf %lf\n",max1,max2,max3);
    fclose(minmax);
    
    fclose(pf);
    
    return massimi;
}

double *min(char *file){
    long long int i, dim = fileLine(file);
    double min1=1E20;
    double min2=1E20;
    double min3=1E20;
    double *minimi = calloc(3,sizeof(double));
    
    FILE *pf = fopen(file,"r");
    
    for(i=0; i<dim; i++){
        fscanf(pf,"%lf %lf %lf\n",minimi,minimi+1,minimi+2);
        
        if(min1 > minimi[0]){
            min1 = minimi[0];
        }
        
        if(min2 > minimi[1]){
            min2 = minimi[1];
        }
        
        if(min3 > minimi[2]){
            min3 = minimi[2];
        }
    }
    
    minimi[0] = min1;
    minimi[1] = min2;
    minimi[2] = min3;
    
    FILE *minmax = fopen("minmax.dat","a+");
    checkPtr(minmax);
    fprintf(minmax,"MINIMI: %lf %lf %lf\n",min1,min2,min3);
    fclose(minmax);
    
    fclose(pf);
    
    return minimi;
}
