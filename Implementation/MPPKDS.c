#include<stdio.h>
#include<stdlib.h>

//Finite Field Index
int p = 257;
//Euler's totient of p
int tp = 256;
//number of noise varibles
int m = 2;
//degree of base polynomials
int n = 2;
//degree of multipler polynomials
int lam = 1;
//upper limit
int ell[2] = { 1, 1 };

int K(int n,int lam,int* ell, int p, int* s, int* v) {
    int tp = p - 1;
    srand(2341); 
    int c [n + 1][(ell[0]+1)*(ell[1]+1)];
    for (int i = 0; i < (n + 1); i++) {
        for(int j = 0; j < ((ell[0]+1)*(ell[1]+1)) ; j++) {
            c[i][j] = (rand() % (tp - 1)) + 0;
        }
    }

    //Print for c 
    /*for (int i = 0; i < (n + 1); i++) {
        for(int j = 0; j < ((ell[0]+1)*(ell[1]+1)) ; j++) {
            //printf("\n%d %d \n", i, j);
            printf("%d ", c[i][j]);
        }
        printf("\n");
    }*/
    int f [lam + 1];
    for (int i = 0; i < (lam + 1); i++) {
        f[i] = (rand() % (tp - 1)) + 0;
    }

    //Print for f
    /*for (int i = 0; i < (lam + 1); i++) {
        printf("%d ", f[i]);
    }
    printf("\n");*/

    int h [lam + 1];
    for (int i = 0; i < (lam + 1); i++) {
        h[i] = (rand() % (tp - 1)) + 0;
    }

    int phi [n + lam + 1][(ell[0]+1)*(ell[1]+1)];
    int psi [n + lam + 1][(ell[0]+1)*(ell[1]+1)];

    for(int i = 0; i < (n + lam + 1); i++) {
        for(int j = 0; j < ((ell[0]+1)*(ell[1]+1)); j++) {
            phi[i][j] = 0;
            psi[i][j] = 0;
        }
    }

    for (int i = 0; i < (n + 1); i++) {   
        for (int j = 0; j < (lam + 1); j++ ) { 
            for(int k = 0; k < ((ell[0]+1)*(ell[1]+1)); k++) {
                phi[i + j][k] = phi[i + j][k] + (c[i][k] * f[j]);
                psi[i + j][k] = psi[i + j][k] + (c[i][k] * h[j]);
            }
        }
    }

    for (int i = 0; i < (n + lam + 1); i++) {
        for(int j = 0; j < ((ell[0]+1)*(ell[1]+1)) ; j++) {
            phi[i][j] = phi[i][j] % tp;
            psi[i][j] = psi[i][j] % tp;
        }
    }

    //Print phi and psi
    /*printf("Phi and Psi:\n");
    for (int i = 0; i < (n + lam + 1); i++) {
        for(int j = 0; j < ((ell[0]+1)*(ell[1]+1)) ; j++) {
            printf("%d ", phi[i][j]);
        }
        printf("\n");
    }

    printf("\n");

    for (int i = 0; i < (n + lam + 1); i++) {
        for(int j = 0; j < ((ell[0]+1)*(ell[1]+1)) ; j++) {
            printf("%d ", psi[i][j]);
        }
        printf("\n");
    }

    printf("\n");*/

    int ephi [n + lam - 1];
    for (int i = 0; i < (n + lam - 1); i++) {
        ephi[i] = (rand() % (tp - 1)) + 0;
    }

    int epsi [n + lam - 1];
    for (int i = 0; i < (n + lam - 1); i++) {
        epsi[i] = (rand() % (tp - 1)) + 0;
    }

    int R0 = ((rand() % ((tp-2)/2) ) + 1) * 2;
    int Rn = ((rand() % ((tp-2)/2) ) + 1) * 2;

    //Print R0, RN, Ephi, and Epsi
    /*printf("R0: %d and RN: %d\n", R0, Rn);

    for(int i = 0; i < (n + lam - 1); i++) {
        printf("%d ", ephi[i]);
    }

    printf("\n");

    for(int i = 0; i < (n + lam - 1); i++) {
        printf("%d ", epsi[i]);
    }

    printf("\n");*/

    int N0 [(ell[0]+1)*(ell[1]+1)];
    int Nn [(ell[0]+1)*(ell[1]+1)];

    for(int i = 0; i < ((ell[0]+1)*(ell[1]+1)); ++i) {
        N0[i] = (R0 * c[0][i]) % tp;
        Nn[i] = (Rn * c[n][i]) % tp;
    }

    //Print N0 and Nn
    /*for(int i = 0; i < ((ell[0]+1)*(ell[1]+1)); ++i) {
        printf("%d ", N0[i]);
    }

    printf("\n");

    for(int i = 0; i < ((ell[0]+1)*(ell[1]+1)); ++i) {
        printf("%d ", Nn[i]);
    }*/

    int P [n + lam - 1][(ell[0]+1)*(ell[1]+1)];
    int Q [n + lam - 1][(ell[0]+1)*(ell[1]+1)];

    for(int i = 0; i < (n + lam - 1); i++) {
        for(int j = 0; j < ((ell[0]+1)*(ell[1]+1)); j++) {
            P[i][j] = 0;
            Q[i][j] = 0;
        }
    }

    for(int i = 0; i < (n + lam - 1); i++) {
        for(int j = 0; j < ((ell[0]+1)*(ell[1]+1)); j++) {
            P[i][j] = 0;
            Q[i][j] = 0;
        }
    }

    for(int i = 0; i < (n + lam - 1); i++) {

        phi[i][0] = phi[i][0] - ephi[i];
        psi[i][0] = phi[i][0] - epsi[i];

        for(int j = 0; j < ((ell[0]+1)*(ell[1]+1)); j++) {
            P[i][j] = R0 * phi[i][j];
            Q[i][j] = Rn * psi[i][j];
        }
    }

    for (int i = 0; i < (n + lam - 1); i++) {
        for(int j = 0; j < ((ell[0]+1)*(ell[1]+1)) ; j++) {
            P[i][j] = P[i][j] % tp;
            Q[i][j] = Q[i][j] % tp;
        }
    }

    //printf("P AND Q: \n");

    /*for (int i = 0; i < (n + lam - 1); i++) {
        for(int j = 0; j < ((ell[0]+1)*(ell[1]+1)) ; j++) {
            printf("%d ", P[i][j]);
        }
        printf("\n");
    }

    printf("\n");

    for (int i = 0; i < (n + lam - 1); i++) {
        for(int j = 0; j < ((ell[0]+1)*(ell[1]+1)) ; j++) {
            printf("%d ", Q[i][j]);
        }
        printf("\n");
    }*/

    int temp [(lam + 1) + (lam + 1) + (1) + (1) + (n + lam - 1) + (n + lam - 1)];

    for(int i = 0; i < (lam + 1); i++) {
        temp[i] = f[i];
    }

    for(int i = 0; i < (lam + 1); i++) {
        temp[i + (lam + 1)] = h[i];
    }

    temp[(lam + 1) + (lam + 1)] = R0;

    temp[(lam + 1) + (lam + 1) + 1] = Rn;

    for(int i = 0; i < (n + lam - 1); ++i) {
        temp[i + (lam + 1) + (lam + 1) + (1) + 1] = ephi[i];
    }

    for(int i = 0; i < (n + lam - 1); ++i) {
        temp[i + (lam + 1) + (lam + 1) + (1) + (1) + (n + lam - 1)] = epsi[i];
    }

    for(int i = 0; i < ((lam + 1) + (lam + 1) + (1) + (1) + (n + lam - 1) + (n + lam - 1)); i++) {
        printf("%d ", temp[i]);
    }

    printf("\n");



    //s =  [f, h, R0,Rn ,Ephi, Epsi]
    //v =  [ list(P),list (Q),list( N0),list(Nn )]

}

int main() {
    int s;
    int v;
    K(n, lam, ell, p, &s, &v);
    return 0;
}
