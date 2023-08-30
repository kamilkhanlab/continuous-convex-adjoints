/* -----------------------------------------------------------------
 * Last Modified by Yulan Zhang
 * 2023/05/15
 * Example 6 for CACE paper
 * Adjoint subgradient evaluation system with reverse-mode subgradient AD
 * For use in comparing CPU time of using reverse AD and forward AD
 * The number of parameters can be modified to test the corresponding computational time
 * -----------------------------------------------------------------*/

 

 /* Include header files for both MC++ and CVODES*/
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <chrono>
#include <cvodes/cvodes.h>             /* prototypes for CVODE fcts., consts.  */
#include <nvector/nvector_serial.h>    /* access to serial N_Vector            */
#include <sunmatrix/sunmatrix_dense.h> /* access to dense SUNMatrix            */
#include <sunlinsol/sunlinsol_dense.h> /* access to dense SUNLinearSolver      */
#include <sundials/sundials_types.h>   /* defs. of realtype, sunindextype      */
#include <sundials/sundials_math.h>    /* definition of ABS */
#include <iostream>
#include <fstream>
#include "interval.hpp"
#include "mccormick.hpp"
#include "reverseMC4.hpp"

using namespace std;


/* Accessor macros */

#define Ith(v,i)    NV_Ith_S(v,i-1)         /* i-th vector component i=1..NEQ */
#define IJth(A,i,j) SM_ELEMENT_D(A,i-1,j-1) /* (i,j)-th matrix component i,j=1..NEQ */

/* Problem Constants */

#define RTOL  RCONST(1e-7)  /* scalar relative tolerance */
#define ATOL  RCONST(1e-7)  /* vector absolute tolerance components */

#define ATOLl    RCONST(1e-7)  /* absolute tolerance for adjoint vars. */
#define ATOLq    RCONST(1e-7)  /* absolute tolerance for quadratures   */

#define T0    RCONST(0.0)   /* initial time */
#define TOUT  RCONST(0.15)   /* final time */
#define TB1   RCONST(0.15)   /* starting point for adjoint problem   */


#define STEPS    150           /* number of steps between check points */
#define NP       300             /* number of problem parameters */
#define NX       1             /* number of state variables */
#define NS       2*NP          /* number of subgradient of each yi */
#define NEQ      NX + NX*4     /* number of equations: original solution, state relaxation, subgradient  */
#define ZERO     RCONST(0.0)
#define xi       0
#define pi       0
#define varyp    -0.5

/* Type : UserData */
/* problem parameters */
typedef mc::Interval I;
typedef mc::McCormick<I> MC;


/* Define parameters*/
double pL[NP];
double pU[NP];
double fixedp[NP];


typedef struct {
    I pI[NP];
    realtype p[NP];
} *UserData;


/* Prototypes of functions by CVODES */
template <typename T, typename U> T Original_RHS(T x[NX], U p[NP], int n);
template <typename T> T Original_initial(T p[NP], int n);
static N_Vector x_initial(N_Vector x, void* user_data);
static int f(realtype t, N_Vector x_Re, N_Vector dx_Re, void* user_data);
static int fB(realtype t, N_Vector x, N_Vector xB, N_Vector xBdot, void* user_dataB);
static int fQB(realtype t, N_Vector x, N_Vector xB, N_Vector qBdot, void* user_dataB);
static N_Vector S_initial(N_Vector xB, N_Vector S_init,void* user_data);
static int ewt(N_Vector x_Re, N_Vector w, void* user_data);


/* Prototypes of private functions */
static void PrintHead(realtype tB0);
static void PrintOutput(realtype tfinal, N_Vector x, N_Vector xB, N_Vector qB);
static void PrintOutput1(realtype time, realtype t, N_Vector x);
static void PrintOutput2(void* cvode_mem, realtype t, N_Vector u);

static int check_retval(void* returnvalue, const char* funcname, int opt);

/*
 *--------------------------------------------------------------------
 * MAIN PROGRAM
 *--------------------------------------------------------------------
 */


int main()
{
    
    typedef std::chrono::high_resolution_clock Time;
    typedef std::chrono::duration<float> fsec;
    auto tstart = Time::now();
    
    SUNContext sunctx;
    
    SUNMatrix A, AB;
    SUNLinearSolver LS, LSB;

    void* cvode_mem;

    UserData data;

    long int nst, nstB;
    realtype time;

    N_Vector x;
    N_Vector xB, qB;
    N_Vector S0;

    int retval, ncheck;
    int steps;
    int indexB;

    MC pMC[NP];

    CVadjCheckPointRec* ckpnt;

    realtype reltolQ, abstolQ;

    realtype reltolB, abstolB, abstolQB;

    data = NULL;
    A = AB = NULL;
    LS = LSB = NULL;
    cvode_mem = NULL;
    ckpnt = NULL;
    x = xB = qB = NULL;
    S0 = NULL;

    /* Initialize user data structure sclar p and interval PI*/
    data = (UserData)malloc(sizeof * data);
    if (check_retval((void*)data, "malloc", 2)) return(1);

    
    /* define the values of paraeters */
    for(int i=0; i<NP; i++){
        pL[i]=-2.0;
        pU[i]=2.0;
        fixedp[i]=1.0;
    }
    
    for (int j = 0; j < NP; j++) {
        if (j==pi){
            data->p[j] = varyp;
        }
        else{
            data->p[j] = fixedp[j];
        }
        data->pI[j] = I(pL[j], pU[j]);
    }

    /* Create the SUNDIALS context that all SUNDIALS objects require */
    retval = SUNContext_Create(NULL, &sunctx);
    if (check_retval(&retval, "SUNContext_Create", 1)) return(1);

    /* Initialize state variables */
    x = N_VNew_Serial(NEQ,sunctx);
    if (check_retval((void*)x, "N_VNew_Serial", 0)) return(1);
    x = x_initial(x, data);


    /* Set the scalar relative and absolute tolerances reltolQ and abstolQ */
    reltolQ = RTOL;
    abstolQ = ATOLq;

    /* Create and allocate CVODES memory for forward run */
    printf("Create and allocate CVODES memory for forward runs\n");

    /* Call CVodeCreate to create the solver memory and specify the
       Backward Differentiation Formula */
    cvode_mem = CVodeCreate(CV_BDF,sunctx);
    if (check_retval((void*)cvode_mem, "CVodeCreate", 0)) return(1);

    /* Call CVodeInit to initialize the integrator memory and specify the
       user's right-hand side function in y'=f(t,y), the initial time T0, and
       the initial dependent variable vector y. */


       //// The first bug, failing to allocate memory.
       //// Solution: check the dimensional of y and dy
    retval = CVodeInit(cvode_mem, f, T0, x);
    if (check_retval(&retval, "CVodeInit", 1)) return(1);


    /* Call CVodeWFtolerances to specify a user-supplied function ewt that sets
       the multiplicative error weights w_i for use in the weighted RMS norm */
    retval = CVodeWFtolerances(cvode_mem, ewt);
    if (check_retval(&retval, "CVodeSetEwtFn", 1)) return(1);

    /* Attach user data */
    retval = CVodeSetUserData(cvode_mem, data);
    if (check_retval(&retval, "CVodeSetUserData", 1)) return(1);


    /* Create dense SUNMatrix */
    A = SUNDenseMatrix(NEQ, NEQ,sunctx);
    if (check_retval((void*)A, "SUNDenseMatrix", 0)) return(1);


    /* Create dense SUNLinearSolver */
    LS = SUNLinSol_Dense(x, A,sunctx);
    if (check_retval((void*)LS, "SUNLinSol_Dense", 0)) return(1);


    /* Attach the matrix and linear solver */
    retval = CVodeSetLinearSolver(cvode_mem, LS, A);
    if (check_retval(&retval, "CVodeSetLinearSolver", 1)) return(1);

    /* Set the maximum number of step size */
    retval = CVodeSetMaxNumSteps(cvode_mem, 2000);
    if (check_retval(&retval, "CVodeSetMaxNumSteps", 1)) return(1);

    /* Allocate global memory */
    /* Call CVodeAdjInit to update the CVODES memory block by allocating the internal
       memory needed for backward integration.*/
    steps = STEPS; /* no. of integration steps between two consecutive checkpoints*/
    retval = CVodeAdjInit(cvode_mem, steps, CV_HERMITE);
    /*
    retval = CVodeAdjInit(cvode_mem, steps, CV_POLYNOMIAL);
    */
    if (check_retval(&retval, "CVodeAdjInit", 1)) return(1);

    /* Perform forward run */
    printf("Forward integration ... ");

    /* Call CVodeF to integrate the forward problem over an interval in time and
       saves checkpointing data */
    retval = CVodeF(cvode_mem, TOUT, x, &time, CV_NORMAL, &ncheck);
    if (check_retval(&retval, "CVodeF", 1)) return(1);
    //retval = CVodeF(cvode_mem, TOUT, y, &time, CV_ONE_STEP, &ncheck);
    //if (check_retval(&retval, "CVodeF", 1)) return(1);


    retval = CVodeGetNumSteps(cvode_mem, &nst);
    if (check_retval(&retval, "CVodeGetNumSteps", 1)) return(1);


    printf("done ( nst = %ld )\n", nst);
    printf("\nncheck = %d\n\n", ncheck);  // (int)the number of(internal) checkpoints stored so far.
    printf("-------------------------------------------------\n\n");

    // Print the solutions of forward sensitivity analysis
    // This could be compared to the results obtained by running Example1.cpp
    printf("Solutions to forward sensitivity analysis\n");
    std::cout << "p1: " << fixedp[0] << std::endl;
    printf("===========================================");
    printf("============================\n");
    printf("     T     Q       H      NST           x\n");
    printf("===========================================");
    printf("============================\n");
    PrintOutput2(cvode_mem, time, x);


    /* Initialize xB */
    /* xB = dg/dx_Re for i = 1,..,NX at T*/
    /* where x_Re = xicv, xicc*/
    xB = N_VNew_Serial(2 * NX,sunctx);
    if (check_retval((void*)xB, "N_VNew_Serial", 0)) return(1);
    for (int i = 0; i < 2 * NX; i++) {
        Ith(xB, i + 1) = ZERO;
    }

    /* Define function g*/
    /* In this case, g is set to x5^cv*/
    Ith(xB, 1) = RCONST(1.0);


    /* Initialize qB */
    /* qB = 0 at T */
    qB = N_VNew_Serial(NP,sunctx);
    if (check_retval((void*)qB, "N_VNew", 0)) return(1);
    for (int i = 0; i < NP; i++) {
        Ith(qB, i + 1) = ZERO;
    }
    

    /* Set the scalar relative tolerance reltolB */
    reltolB = RTOL;

    /* Set the scalar absolute tolerance abstolB */
    abstolB = ATOLl;

    /* Set the scalar absolute tolerance abstolQB */
    abstolQB = ATOLq;

    /* Create and allocate CVODES memory for backward run */
    printf("-------------------------------------------------\n");
    printf("Create and allocate CVODES memory for backward run\n");


    /* Call CVodeCreateB to specify the solution method for the backward
       problem. */
    retval = CVodeCreateB(cvode_mem, CV_BDF, &indexB);
    if (check_retval(&retval, "CVodeCreateB", 1)) return(1);

    /* Call CVodeInitB to allocate internal memory and initialize the
       backward problem. */
    retval = CVodeInitB(cvode_mem, indexB, fB, TB1, xB);
    if (check_retval(&retval, "CVodeInitB", 1)) return(1);

    /* Set the scalar relative and absolute tolerances. */
    retval = CVodeSStolerancesB(cvode_mem, indexB, reltolB, abstolB);
    if (check_retval(&retval, "CVodeSStolerancesB", 1)) return(1);

    /* Attach the user data for backward problem. */
    retval = CVodeSetUserDataB(cvode_mem, indexB, data);
    if (check_retval(&retval, "CVodeSetUserDataB", 1)) return(1);

    /* Create dense SUNMatrix for use in linear solves */
    // Dimension of AB is set to (2*NY, 2*NY)
    AB = SUNDenseMatrix(2 * NX, 2 * NX,sunctx);
    if (check_retval((void*)AB, "SUNDenseMatrix", 0)) return(1);

    /* Create dense SUNLinearSolver object */
    LSB = SUNLinSol_Dense(xB, AB,sunctx);
    if (check_retval((void*)LSB, "SUNLinSol_Dense", 0)) return(1);


    /* Attach the matrix and linear solver */
    retval = CVodeSetLinearSolverB(cvode_mem, indexB, LSB, AB);
    if (check_retval(&retval, "CVodeSetLinearSolverB", 1)) return(1);

    /* Call CVodeQuadInitB to allocate internal memory and initialize backward
   quadrature integration. */
    retval = CVodeQuadInitB(cvode_mem, indexB, fQB, qB);
    if (check_retval(&retval, "CVodeQuadInitB", 1)) return(1);

    /* Call CVodeSetQuadErrCon to specify whether or not the quadrature variables
       are to be used in the step size control mechanism within CVODES. Call
       CVodeQuadSStolerances or CVodeQuadSVtolerances to specify the integration
       tolerances for the quadrature variables. */
    retval = CVodeSetQuadErrConB(cvode_mem, indexB, SUNTRUE);
    if (check_retval(&retval, "CVodeSetQuadErrConB", 1)) return(1);

    /* Call CVodeQuadSStolerancesB to specify the scalar relative and absolute tolerances
       for the backward problem. */
    retval = CVodeQuadSStolerancesB(cvode_mem, indexB, reltolB, abstolQB);
    if (check_retval(&retval, "CVodeQuadSStolerancesB", 1)) return(1);

    /* Backward Integration */

    PrintHead(TB1);


    /* Then get results at t = T0*/

    //retval = CVodeB(cvode_mem, T0, CV_NORMAL);
    //if (check_retval(&retval, "CVodeB", 1)) return(1);

    retval = CVodeB(cvode_mem, T0, CV_NORMAL);
    if (check_retval(&retval, "CVodeB", 1)) return(1);


    CVodeGetNumSteps(CVodeGetAdjCVodeBmem(cvode_mem, indexB), &nstB);
    printf("Done ( nst = %ld )\n", nstB);

    retval = CVodeGetB(cvode_mem, indexB, &time, xB);
    if (check_retval(&retval, "CVodeGetB", 1)) return(1);

    /* Call CVodeGetQuadB to get the quadrature solution vector after a
   successful return from CVodeB. */
    retval = CVodeGetQuadB(cvode_mem, indexB, &time, qB);
    if (check_retval(&retval, "CVodeGetQuadB", 1)) return(1);

    retval = CVodeGetAdjY(cvode_mem, T0, x);
    if (check_retval(&retval, "CVodeGetAdjY", 1)) return(1);

    
    PrintOutput(time, x, xB, qB);


    /* Free memory */
    printf("Free memory\n\n");

    CVodeFree(&cvode_mem);
    N_VDestroy(x);
    N_VDestroy(xB);
    N_VDestroy(qB);
    SUNLinSolFree(LS);
    SUNMatDestroy(A);
    SUNLinSolFree(LSB);
    SUNMatDestroy(AB);
    SUNContext_Free(&sunctx);                 /* Free the SUNDIALS context */
    
    if (ckpnt != NULL) free(ckpnt);
    free(data);
    
    auto tend = Time::now();
    
    fsec s_float = tend - tstart;
    std::cout<< s_float.count() <<"s\n";

    return(0);

}

/*
 *--------------------------------------------------------------------
 * FUNCTIONS CALLED BY CVODES
 *--------------------------------------------------------------------
 */

 /*
  * Initial conditions for user-supplied ODE.
  * Set initial values for original solution, state bounds and state relaxations, denoted as vector x,
 */


template <typename T> T Original_initial(T p[NP], int n)
{

    T x0;

    /* Size of n depends on the number of functions in user-supplied ODE system*/
    switch (n)
    {
    case 0:
        x0 = -2 + 0.0 * p[0];
        break;
    }
    
    return x0;

}


/*
* f routine, which is the original right-hand side function returning the interval or McCormick.
*/

template <typename T, typename U> T Original_RHS(T x[NX], U p[NP], int n)
{
    T f_rhs;
    
    switch (n)
    {
    case 0:
        
        f_rhs = 0;
            
        for (int i=0; i<NP;i++){
                
            f_rhs = f_rhs + sin(p[i]/2)*x[0];
        }
            
        break;
    }
    
    return f_rhs;
}




/*
 * Set initial conditions for an auxiliary system which solves original ODE solutions,
 * along with convex/concave relaxations.
*/

static N_Vector x_initial(N_Vector x, void* user_data)
{
    MC x0MC[NX], pMC[NP];
    realtype p[NP], x0Aug[NX];

    UserData data;
    data = (UserData)user_data;

    /* Assign values to p and MC pMC*/
    for (int j = 0; j < NP; j++) {
        p[j] = data->p[j];
        pMC[j] = MC(I(pL[j], pU[j]), p[j]);
    }



    /* Initial conditions for ODE system*/
    for (int j = 0; j < NX; j++) {

        x0Aug[j] = Original_initial(p, j);
        x0MC[j] = Original_initial(pMC, j);
    }

    /* Construct x vector*/
    //x = N_VNew_Serial(NEQ);


    for (int j = 0; j < NX; j++) {
        Ith(x, j + 1 + 0 * NX) = x0Aug[j];
        Ith(x, j + 1 + 1 * NX) = x0MC[j].l();
        Ith(x, j + 1 + 2 * NX) = x0MC[j].u();
        Ith(x, j + 1 + 3 * NX) = x0MC[j].cv();
        Ith(x, j + 1 + 4 * NX) = x0MC[j].cc();
    }

    return x;

}



/*
* RHS of the auxiliary ODE system which solves original ODE solutions,
* along with convex/concave relaxations.
* This ODE is solved in a forward mode.
*/

static int f(realtype t, N_Vector x, N_Vector dx, void* user_data)
{

    MC pMC[NP], xMC[NX];
    I pI[NP], xI[NX];
    realtype dxL[NX], dxU[NX], dxcv[NX], dxcc[NX];
    realtype xL[NX], xU[NX], xcv[NX], xcc[NX];
    realtype p[NP];
    realtype xori[NX], xd[NX];
    UserData data;
    data = (UserData)user_data;


    /* Assign values to p, interval pI, McCormick pMC */
    for (int j = 0; j < NP; j++) {
        p[j] = data->p[j];
        pI[j] = I(pL[j], pU[j]);
        pMC[j] = MC(I(pL[j], pU[j]), p[j]);
    }


    /* Generate x_ori, xL, xU, xcv and xcc arrays using values from vector x*/
    for (int j = 0; j < NX; j++) {
        xori[j] = Ith(x, j + 1 + 0 * NX);
        xL[j] = Ith(x, j + 1 + 1 * NX);
        xU[j] = Ith(x, j + 1 + 2 * NX);
        xcv[j] = Ith(x, j + 1 + 3 * NX);
        xcc[j] = Ith(x, j + 1 + 4 * NX);
    }


    /* Initialize interval xI, McCormick xMC */
    for (int j = 0; j < NX; j++) {
        xI[j] = I(xL[j], xU[j]);
        xMC[j] = MC(I(xL[j], xU[j]), xcv[j], xcc[j]);
    }


    /* Computation for RHS of the auxiliary ODEs*/
    for (int j = 0; j < NX; j++) {

        /*-------------------------------------------------------------*/
        /*-------------------------------------------------------------*/
        /* Construct the original ODE system's RHS */
        xd[j] = Original_RHS(xori, p, j);


        /*-------------------------------------------------------------*/
        /*-------------------------------------------------------------*/
        /* Construct the state bounds computation system's RHS */

        /* Flatten the ith interval xI (xiL, xiU) to (xiL, xiL)*/
        xI[j] = I(xL[j], xL[j]);
        /* Apply the flattened xI into the original RHS function, then obtain the lower bound */
        dxL[j] = Original_RHS(xI, pI, j).l();


        /* Flatten the ith interval xI (xiL, xiU) to (xiU, xiU)*/
        xI[j] = I(xU[j], xU[j]);
        /* Apply the flattened xI into the original RHS function, then obtain the upper bound */
        dxU[j] = Original_RHS(xI, pI, j).u();


        /* Unflatten the ith interval*/
        xI[j] = I(xL[j], xU[j]);


        /*-------------------------------------------------------------*/
        /*-------------------------------------------------------------*/
        /* Construct the state relaxations computation system's RHS */

        /* Flatten the ith xMC (xiL,xiU,xicv,xicc) to (xiL,xiU,xicv,xicv)*/
        xMC[j] = MC(I(xL[j], xU[j]), xcv[j], xcv[j]);
        /* Apply the flattened xMC into the original RHS function,
           then obtain the convex relaxation */
        dxcv[j] = Original_RHS(xMC, pMC, j).cv();


        /* Flatten the ith xMC (xiL,xiU,xicv,xicc) to (xiL,xiU,xicc,xicc)*/
        xMC[j] = MC(I(xL[j], xU[j]), xcc[j], xcc[j]);
        /* Apply the flattened xMC into the original RHS function,
           then obtain the concave relaxation */
        dxcc[j] = Original_RHS(xMC, pMC, j).cc();


        /* Unflatten the ith xMC*/
        xMC[j] = MC(I(xL[j], xU[j]), xcv[j], xcc[j]);

        /*-------------------------------------------------------------*/
        /*-------------------------------------------------------------*/
        /* Construct dx vextor*/


        Ith(dx, j + 1 + 0 * NX) = xd[j];
        Ith(dx, j + 1 + 1 * NX) = dxL[j];
        Ith(dx, j + 1 + 2 * NX) = dxU[j];
        Ith(dx, j + 1 + 3 * NX) = dxcv[j];
        Ith(dx, j + 1 + 4 * NX) = dxcc[j];


    }

    return(0);
}

/*
 * EwtSet function. Computes the error weights at the current solution.
 */

static int ewt(N_Vector x, N_Vector w, void* user_data)
{


    int i;
    realtype xx, ww, rtol, atol[NEQ];

    rtol = RTOL;

    for (int j = 0; j < NEQ; j++) {
        atol[j] = ATOL;
    }

    for (i = 1; i <= NEQ; i++) {
        xx = Ith(x, i);
        ww = rtol * SUNRabs(xx) + atol[i - 1];
        if (ww <= 0.0) return (-1);
        Ith(w, i) = 1.0 / ww;
    }

    return(0);
}

/*
 * fB routine. Compute fB(t,x,xB).
 * fB is the RHS function for ODE system which solves lambda.
 * fB = -lambda^T * df/dx
*/


static int fB(realtype t, N_Vector x, N_Vector xB, N_Vector xBdot, void* user_dataB)
{
    
    SUNContext sunctx;
    SUNContext_Create(NULL, &sunctx);
    
    UserData data;
    data = (UserData)user_dataB;

    realtype p[NP], xcv[NX], xcc[NX], xL[NX], xU[NX];
    MC xMCsub[NX], pMCsub[NP];
    realtype l[2 * NX];
    double tempsub[2 * NX * 2 * NX];


    /* Assign values to p, and MC pMCsub (for subgradient evaluations) */
    /* Initialize subgradeints for pMCsub with respect to relaxed x*/
    /* Since p are not functions of x, dp/dx should be equal to 0
       and the dimension of dp/dx should be the same as df/dx*/
    for (int j = 0; j < NP; j++) {
        p[j] = data->p[j];
        pMCsub[j] = MC(I(pL[j], pU[j]), p[j]);
        pMCsub[j].sub(NRev);
    }


    /* Initialize vector lambda with size of 2*NX */
    for (int j = 0; j < 2 * NX; j++) {
        l[j] = Ith(xB, j + 1);
    }


    /* Collect xL, xU, xcv, xcc values from vector x */
    for (int j = 0; j < NX; j++) {
        xL[j] = Ith(x, j + 1 + 1 * NX);
        xU[j] = Ith(x, j + 1 + 2 * NX);
        xcv[j] = Ith(x, j + 1 + 3 * NX);
        xcc[j] = Ith(x, j + 1 + 4 * NX);
    }


    /* Initialize McCormick xMCsub */
    for (int j = 0; j < NX; j++)
    {
        xMCsub[j] = MC(I(xL[j], xU[j]), xcv[j], xcc[j]);
        xMCsub[j].sub(NRev);
    }


    /* Initialize subgradients for xMCsub with 1 or 0 */
    double ini_sub[NRev * NRev] = {1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1};

    /*-------------------------------------------------------------*/
    /*-------------------------------------------------------------*/
    /* Derivatives (df/dx) computation*/
    /* Debug: Flatten the convex and concave relaxations*/

    for (int j = 0; j < NX; j++) {

        /* Flatten the ith xMCsub (xiL,xiU,xicv,xicc)
        to (xiL,xiU,xicv,xicv)*/
        xMCsub[j] = MC(I(xL[j], xU[j]), xcv[j], xcv[j]);

        /* Apply the flattened xMCsub into fRevAD_dfdx function */
        /* Compute the subgradients in reverse mode */
        N_Vector adcvcc_flcv = N_VNew_Serial(4 * NX, sunctx);
        
        adcvcc_flcv = fRevAD_dfdx(xMCsub, pMCsub, ini_sub, j, 0);

        /* Flattening the ith xMCsub (xiL,xiU,xicv,xicc)
        to (xiL,xiU,xicc,xicc)*/
        xMCsub[j] = MC(I(xL[j], xU[j]), xcc[j], xcc[j]);


        /* Apply the flattened the xMCsub into fRevAD_dfdx function */
        /* Compute the subgradients in reverse mode */
        N_Vector adcvcc_flcc = N_VNew_Serial(4 * NX,sunctx);
        adcvcc_flcc = fRevAD_dfdx(xMCsub, pMCsub, ini_sub, j, 4);


        /* Unflattening the ith xMCsub*/
        xMCsub[j] = MC(I(xL[j], xU[j]), xcv[j], xcc[j]);

        /* Storing values for constructing the rhs functions*/
        for (int i = 0; i < 2 * NX; i++)
        {
            tempsub[2 * j * 2 * NX + i] = Ith(adcvcc_flcv, 1 + i);

            tempsub[(2 * j + 1) * 2 * NX + i] = Ith(adcvcc_flcc, 2 * NX + 1 + i);
        }

    }


    double tempxbdot[2 * NX];
    for (int i = 0; i < 2 * NX; i++)
    {
        tempxbdot[i] = 0;

        for (int j = 0; j < 2 * NX; j++)
        {

            tempxbdot[i] = tempxbdot[i] + tempsub[i + j * 2 * NX] * l[j];

        }

        Ith(xBdot, i + 1) = -tempxbdot[i];
    }

    SUNContext_Free(&sunctx);                 /* Free the SUNDIALS context */
    return(0);

}



/*
 * fQB routine. Compute integrand for quadratures
 * fQB is the RHS function for the ODE system which is to compute the integration of lambda^T*df/dp.
*/

static int fQB(realtype t, N_Vector x, N_Vector xB, N_Vector qBdot, void* user_dataB)
{
    
    SUNContext sunctx;
    SUNContext_Create(NULL, &sunctx);
    N_Vector adcvcc_flcv;
    N_Vector adcvcc_flcc;

    UserData data;
    data = (UserData)user_dataB;
    realtype  theta_B[2 * NX * NP];
    realtype l[2 * NX];
    MC pMCsub[NP], xMCsub[NX];
    realtype p[NP];
    realtype xL[NX], xU[NX], xcv[NX], xcc[NX];
 
 
    /* Assign values to p, interval pI, and MC pMCsub (for subgradients evaluations)*/
    for (int j = 0; j < NP; j++) {
        p[j] = data->p[j];
        pMCsub[j] = MC(I(pL[j], pU[j]), p[j]);
    }
 
 
    /* Initialize subgradeints for pMCsub with respect to p*/
    double ini_sub[NRev * NRev] = {1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1 };
    
 
 
    /* Collect values from x */
    for (int j = 0; j < NX; j++) {
        xL[j] = Ith(x, j + 1 + 1 * NX);
        xU[j] = Ith(x, j + 1 + 2 * NX);
        xcv[j] = Ith(x, j + 1 + 3 * NX);
        xcc[j] = Ith(x, j + 1 + 4 * NX);
    }
 
 
    /* Initialize McCormick xMCsub */
    /* Initialize subgradients for xMCsub with respect to p*/
    /* When calculating the partial derivatives df/dp, we do not consider dx/dp*/
    /* Comments from Dr.Khan: I think it's NOT supposed to include the chain-rule-y dependence of x on p,
    since that's included separately in the adjoint ODEs already.*/
    for (int j = 0; j < NX; j++) {
        xMCsub[j] = MC(I(xL[j], xU[j]), xcv[j], xcc[j]);
        xMCsub[j].sub(NRev);
    }
 
 
    /*-------------------------------------------------------------*/
    /*-------------------------------------------------------------*/
    /* Derivatives (df/dp) computation*/
 
    for (int j = 0; j < NX; j++) {
 
        /* Flattening the ith xMCsub (xiL,xiU,xicv,xicc,xicvsub,xiccsub)
        to (xiL,xiU,xicv,xicv,xicvsub,xicvsub)*/
        xMCsub[j] = MC(I(xL[j], xU[j]), xcv[j], xcv[j]);
        xMCsub[j].sub(NRev);
 
 
        /* Apply the flattened xMCsub into fRevAD_dfdp function */
        /* Compute the subgradients in reverse mode */
        adcvcc_flcv = N_VNew_Serial(2 * NP,sunctx);
        adcvcc_flcv = fRevAD_dfdp(xMCsub, pMCsub, ini_sub, j);
 
 
        /* Flattening the ith xMCsub (xiL,xiU,xicv,xicc,xicvsub,xiccsub)
        to (xiL,xiU,xicv,xicv,xicvsub,xicvsub)*/
        xMCsub[j] = MC(I(xL[j], xU[j]), xcc[j], xcc[j]);
        xMCsub[j].sub(NRev);
 
 
        /* Apply the flattened xMCsub into fRevAD_dfdp function */
        /* Compute the subgradients in reverse mode */
        adcvcc_flcc = N_VNew_Serial(2 * NP,sunctx);
        adcvcc_flcc = fRevAD_dfdp(xMCsub, pMCsub, ini_sub, j);
 
 
        /* Unflattening the ith xMCsub*/
        xMCsub[j] = MC(I(xL[j], xU[j]), xcv[j], xcc[j]);
        xMCsub[j].sub(NRev);
 
        /* Store the values to array theta_B*/
        for (int i = 0; i < NP; i++)
        {
            theta_B[2 * j * NP + i] = Ith(adcvcc_flcv, i + 1);
 
            theta_B[(2 * j + 1) * NP + i] = Ith(adcvcc_flcc, NP + i + 1);
        }
 
    }
 
 
    /* Recall values to lambda calculated by fB */
    for (int j = 0; j < 2 * NX; j++) {
 
        l[j] = Ith(xB, j + 1);
    }
 
 
    /* Construct the rhs functions */
 
    double tempqbdot[NP];
    for (int i = 0; i < NP; i++)

    {
        tempqbdot[i] = 0;
        for (int j = 0; j < 2 * NX; j++)
        {
            tempqbdot[i] = tempqbdot[i] + theta_B[i + j * NP] * l[j];
        }
 
        Ith(qBdot, i + 1) = tempqbdot[i];
    }

    return(0);
 
}



/*
 *--------------------------------------------------------------------
 * PRIVATE FUNCTIONS
 *--------------------------------------------------------------------
 */

/*
   * Print heading for backward integration
*/


static void PrintHead(realtype tB0)
{
    printf("Backward integration from tB0 = %12.4e\n\n", tB0);
}


/*
 * Print intermediate results during backward integration
 */

static void PrintOutput1(realtype time, realtype t, N_Vector x)
{
    printf("--------------------------------------------------------\n");
    printf("returned t: %12.4e\n", time);
    printf("tout:       %12.4e\n", t);
    printf("y(t):       %12.4e %12.4e \n",
        Ith(x, 1), Ith(x, 2));
    printf("--------------------------------------------------------\n\n");
}

/*
 * Print current t, step count, order, stepsize, and solution for forward sensitivity analysis.
 */

static void PrintOutput2(void* cvode_mem, realtype t, N_Vector u)
{
    long int nst;  // number of steps taken by cvodes
    int qu, retval; // qu - integration method order used on the last internal step
    realtype hu, * udata; // hu - step size taken on the last internal step

    
    /* capturing a returned array/pointer */
    udata = N_VGetArrayPointer(u);

    retval = CVodeGetNumSteps(cvode_mem, &nst); // Cumulative number of internal steps
    check_retval(&retval, "CVodeGetNumSteps", 1);
    retval = CVodeGetLastOrder(cvode_mem, &qu); // integration method order used during the last step
    check_retval(&retval, "CVodeGetLastOrder", 1);
    retval = CVodeGetLastStep(cvode_mem, &hu); // Step size used for the last step
    check_retval(&retval, "CVodeGetLastStep", 1);


    printf("%8.3e %2d  %8.3e %5ld\n", t, qu, hu, nst);

    printf("                  Original Solution ");
    printf("%12.4e \n", udata[xi + 0 * NX]);

    printf("                  Lower Bound       ");
    printf("%12.4e  \n", udata[xi + 1 * NX]);


    printf("                  Upper Bound       ");
    printf("%12.4e \n", udata[xi + 2 * NX]);

    printf("                  Convex Relaxation ");
    printf("%12.4e \n", udata[xi + 3 * NX]);

    printf("                  Concave Relaxation");
    printf("%12.4e \n", udata[xi + 4 * NX]);
    
    
}

/*
 * Print final results of backward integration
 */

static void PrintOutput(realtype tfinal, N_Vector x_Re, N_Vector xB, N_Vector qB)
{
    
    printf("--------------------------------------------------------\n");
    printf("returned t: %12.4e\n", tfinal);
    printf("lambda(T):  %12.4e %12.4e \n",
        Ith(xB, 9), Ith(xB, 10));
    printf("x(t0):      %12.4e %12.4e \n",
        Ith(x_Re, 9), Ith(x_Re, 10));
    printf("--------------------------------------------------------\n");
    printf("Sensitivity       p1           p2           p3           p4\n\n");
    printf("dg/dp:      %12.4e %12.4e %12.4e %12.4e \n" ,
        -Ith(qB, 1), -Ith(qB, 2), -Ith(qB, 3), -Ith(qB, 4));
    printf("                  p5           p6           p7           p8\n\n");
    printf("dg/dp:      %12.4e %12.4e %12.4e %12.4e \n" ,
        -Ith(qB, 5), -Ith(qB, 6) , -Ith(qB, 7), -Ith(qB, 8));
    printf("--------------------------------------------------------\n\n");
    
}


/*
 * Check function return value.
 *    opt == 0 means SUNDIALS function allocates memory so check if
 *             returned NULL pointer
 *    opt == 1 means SUNDIALS function returns an integer value so check if
 *             retval < 0
 *    opt == 2 means function allocates memory so check if returned
 *             NULL pointer
 */
static int check_retval(void* returnvalue, const char* funcname, int opt)
{
    int* retval;

    /* Check if SUNDIALS function returned NULL pointer - no memory allocated */
    if (opt == 0 && returnvalue == NULL) {
        fprintf(stderr, "\nSUNDIALS_ERROR: %s() failed - returned NULL pointer\n\n",
            funcname);
        return(1);
    }

    /* Check if retval < 0 */
    else if (opt == 1) {
        retval = (int*)returnvalue;
        if (*retval < 0) {
            fprintf(stderr, "\nSUNDIALS_ERROR: %s() failed with retval = %d\n\n",
                funcname, *retval);
            return(1);
        }
    }

    /* Check if function returned NULL pointer - no memory allocated */
    else if (opt == 2 && returnvalue == NULL) {
        fprintf(stderr, "\nMEMORY_ERROR: %s() failed - returned NULL pointer\n\n",
            funcname);
        return(1);
    }

    return(0);
}
