#ifndef REVERSEAD_H
#define REVERSEAD_H
#include <vector>
#include <iostream>
#include <algorithm>
#include <nvector/nvector_serial.h>    /* access to serial N_Vector            */
#include "interval.hpp"                /* access to interval                   */
#include "mccormick.hpp"               /* access to McCormick relaxations      */
#include "revmccormick.hpp"            /* access to McCormick relaxations with reverse subgradients */
    
/* Accessor macros */
#define Ith(v,i)    NV_Ith_S(v,i-1)         /* i-th vector component i=1..NEQ */
#define IJth(A,i,j) SM_ELEMENT_D(A,i-1,j-1) /* (i,j)-th matrix component i,j=1..NEQ */

#define NP       8           /* number of problem parameters */
#define NX       5           /* number of state variables */
#define NRev     4           /* total number of cvsub/ccsub of each component in reverse MC*/
#define L1       54          /* tape length */
#define L2       54          /* tape length */
#define L3       58          /* tape length */
#define L4       61          /* tape length */
#define L5       60          /* tape length */
#define L        61
typedef mc::Interval I;
typedef mc::McCormick<I> MC;
typedef mc::RevMcCormick<MC> RevMC;
        
N_Vector fRevAD_dfdp(MC xMC[NX], MC pMC[NP], double sub[NRev * NRev], int n);
N_Vector fRevAD_dfdx(MC xMC[NX], MC pMC[NP], double sub[NRev * NRev], int n, int k);
MC tmpv(double* vsub, int i);
#endif

