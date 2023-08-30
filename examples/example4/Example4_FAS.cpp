/* -----------------------------------------------------------------
 * Last Modified by Yulan Zhang
 * 2023/05/15
 * Example 6 for CACE paper
 * Forward subgradient evaluation system
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

using namespace std;


/* Accessor macros */

#define Ith(v,i)    NV_Ith_S(v,i-1)         /* i-th vector component i=1..NEQ */
#define IJth(A,i,j) SM_ELEMENT_D(A,i-1,j-1) /* (i,j)-th matrix component i,j=1..NEQ */

/* Problem Constants */

#define RTOL  RCONST(1e-7)  /* scalar relative tolerance */
#define ATOL  RCONST(1e-7)  /* vector absolute tolerance components */

#define epsilon RCONST(0)

#define T0    RCONST(0.0)    /* initial time */
#define T1    RCONST(0.15)   /* first output time */
#define TMULT RCONST(1.0)  /* output time factor */
#define NOUT      1   /* number of output times */

#define NP    500             /* number of problem parameters */
#define NX    1             /* number of state variables */
#define NS    2*NP          /* number of subgradients of ith yicv and yicc with respect to each pi (from i=1 to i=NP) */
#define NEQ   NX + NX*4 + NP*2*NX  /* number of equations: original solution, state relaxation, subgradient  */
#define ZERO  RCONST(0.0)
#define xi       0
#define pi       0
#define varyp    -0.5


/* define the values of parameters */
//double pL[NP] = { -10, -5, 0, 0, -1,- 1,0,0,};      /* lower bound of parameters */
//double pU[NP] = { 10, 0, 3, 3, 1,2, 5,5,};  /* upper bound of parameters */
//double fixedp[NP] = {-1,-4.2,2.1,2.9,0.1,-0.1,0.5,0.5 };    /* fixed value of parameters*/
//double pL[NP] = {1.0,-2,-2,-2,-2,-2,-2,-2};
//double pU[NP] = {2,2,2,2,2,2,2,2};
//double fixedp[NP] = {1.2, 1.5, 0.2, 0.2, 0.5, -1, -0.5, -2};
double pL[NP];
double pU[NP];
double fixedp[NP];


/* problem parameters */
typedef mc::Interval I;
typedef mc::McCormick<I> MC;

typedef struct {
    I pI[NP];
    realtype p[NP];
} *UserData;


/* Prototypes of functions by CVODES */
template <typename T, typename U> T Original_RHS(T x[NX], U p[NP], int n);
template <typename T> T Original_initial(T p[NP], int n);
static N_Vector x_initial(N_Vector x, void* user_data);
static int f(realtype t, N_Vector x_Re, N_Vector dx_Re, void* user_data);
static int ewt(N_Vector x_Re, N_Vector w, void* user_data);
static void Getresults(N_Vector u);

/* Prototypes of private functions */
static void PrintOutput(void* cvode_mem, realtype t, N_Vector u);
static int check_retval(void* returnvalue, const char* funcname, int opt);
static void PrintFinalStats(void* cvode_mem);

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
    
    printf("\n\n");
    printf("The results of parametric solution x with respect to p");
    
    /* define the values of parameters */
    for(int i=0; i<NP; i++){
        pL[i]=-2.0;
        pU[i]=2.0;
        fixedp[i]=1.0;
    }

    for (realtype i = pL[0]; i <= pL[0] ; i += 0.05) {
        
        
        SUNContext sunctx;

        SUNMatrix A;
        SUNLinearSolver LS;
        void* cvode_mem;

        UserData data;

        realtype t, tout;
        N_Vector x, abstol;
        int iout, retval;

        MC pMC[NP];


        cvode_mem = NULL;
        data = NULL;
        x = abstol = NULL;
        A = NULL;
        LS = NULL;

        /* User data structure p and pI */
        data = (UserData)malloc(sizeof * data);
        if (check_retval((void*)data, "malloc", 2)) return(1);

        
        for (int j = 0; j < NP; j++) {
            if (j==pi){
                data->p[j] = varyp;
            }
            else{
                data->p[j] = fixedp[j];
            }
            data->pI[j] = I(pL[j], pU[j]);
        }
        /*
        for (int j = 0; j < NP; j++) {
            data->p[j] = fixedp[j];
            data->pI[j] = I(pL[j], pU[j]);
        }*/
        
        
        /* Create the SUNDIALS context that all SUNDIALS objects require */
        retval = SUNContext_Create(NULL, &sunctx);
        if (check_retval(&retval, "SUNContext_Create", 1)) return(1);
        
        /* Allocate and set initial states */
        x = N_VNew_Serial(NEQ,sunctx);
        if (check_retval((void*)x, "N_VNew_Serial", 0)) return(1);
        x = x_initial(x, data);

        /* Create CVODES object */
        cvode_mem = CVodeCreate(CV_BDF, sunctx);
        if (check_retval((void*)cvode_mem, "CVodeCreate", 0)) return(1);


        /* Allocate space for CVODES */
        retval = CVodeInit(cvode_mem, f, T0, x);
        if (check_retval(&retval, "CVodeInit", 1)) return(1);

        /* Use private function to compute error weights */
        retval = CVodeWFtolerances(cvode_mem, ewt);
        if (check_retval(&retval, "CVodeSetEwtFn", 1)) return(1);


        /* Attach user data */
        retval = CVodeSetUserData(cvode_mem, data);
        if (check_retval(&retval, "CVodeSetUserData", 1)) return(1);


        /* Create dense SUNMatrix */
        A = SUNDenseMatrix(NEQ, NEQ, sunctx);
        if (check_retval((void*)A, "SUNDenseMatrix", 0)) return(1);


        /* Create dense SUNLinearSolver */
        LS = SUNLinSol_Dense(x, A, sunctx);
        if (check_retval((void*)LS, "SUNLinSol_Dense", 0)) return(1);


        /* Attach the matrix and linear solver */
        retval = CVodeSetLinearSolver(cvode_mem, LS, A);
        if (check_retval(&retval, "CVodeSetLinearSolver", 1)) return(1);

        /* Set the maximum number of step size */
        retval = CVodeSetMaxNumSteps(cvode_mem, 5000);
        if (check_retval(&retval, "CVodeSetMaxNumSteps", 1)) return(1);


        /* In loop over output points, call CVode, print results, test for error */

        printf("\n\n");
        std::cout << "p1: " << data->p[pi] << std::endl;
        printf("===========================================");
        printf("============================\n");
        printf("     T     Q       H      NST           x\n");
        printf("===========================================");
        printf("============================\n");


        for (iout = 1, tout = T1; iout <= NOUT; iout++, tout *= TMULT) {

            /* Call CVode to get the solution of the IVP problem*/

            /*CV_NORMAL, CV_ONE_STEP*/
            retval = CVode(cvode_mem, tout, x, &t, CV_NORMAL);
            if (check_retval(&retval, "CVode", 1)) break;

            PrintOutput(cvode_mem, t, x);
            printf("-----------------------------------------");
            printf("------------------------------\n");

            //Getresults(x);

        }

        /* Print final statistics */
        PrintFinalStats(cvode_mem);

        /* Free memory */
        N_VDestroy(x);                    /* Free y vector */
        free(data);                              /* Free user data */
        CVodeFree(&cvode_mem);                   /* Free CVODES memory */
        SUNLinSolFree(LS);                       /* Free the linear solver memory */
        SUNMatDestroy(A);                        /* Free the matrix memory */
        SUNContext_Free(&sunctx);                 /* Free the SUNDIALS context */
        //return(0);

    }
    
    auto tend = Time::now();
    
    fsec s_float = tend - tstart;
    std::cout<< s_float.count() <<"s\n";

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
            
            
            //f_rhs = sin(p[0]/2)*x[0] + sin(p[1]/2)*x[0];
            //f_rhs = exp(f_rhs);
            
            //p[0]*(pow(x[0],2)-1) + x[0]*p[1] + sin(p[5]/5)*x[0] + sin(p[6]/6) + cos(p[7]/7)+ p[4]*x[0] + x[0]*(p[2]-0.2) + pow(p[3],2) + p[8]*p[9]*x[0];// +p[3]*p[4];
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


    /* Initialize subgradeints for pMC with respect to p themselves*/
    // e.g. if NP = 2, cvsub_p1 = 1.0, 0.0, ccsub_p1 = 1.0, 0.0
    double sub[NP * NP] = { 0 };
    for (int j = 0; j < NP * NP; j++) {

        for (int i = 0; i < NP; i++) {

            if (j == i + i * NP) {

                sub[j] = 1.0;
            }
        }

    }

    /* Set subgradients for pMC*/
    for (int j = 0; j < NP; j++) {

        pMC[j].sub(NP, &sub[j * NP], &sub[j * NP]);
    }

    /* Initial conditions for ODE system*/
    for (int j = 0; j < NX; j++) {

        x0Aug[j] = Original_initial(p, j);
        x0MC[j] = Original_initial(pMC, j);
    }

    /* Construct x vector*/
    //x = N_VNew_Serial(NEQ);

    /* Initialize x at t0*/
    /* x = x_original, lower bounds, upper bounds, convex relaxations, concave relaxations. */

    for (int j = 0; j < NX; j++) {
        Ith(x, j + 1 + 0 * NX) = x0Aug[j];
        Ith(x, j + 1 + 1 * NX) = x0MC[j].l();
        Ith(x, j + 1 + 2 * NX) = x0MC[j].u();
        Ith(x, j + 1 + 3 * NX) = x0MC[j].cv();
        Ith(x, j + 1 + 4 * NX) = x0MC[j].cc();

        for (int i = 0; i < NP; i++) {

            Ith(x, 1 + 5 * NX + j * NP + i) = x0MC[j].cvsub(i); //dxjcv/dpi


            Ith(x, 1 + 5 * NX + NX * NP + j * NP + i) = x0MC[j].ccsub(i); //dxjcc/dpi
        }
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

    //Forward sensitivity analysis
    realtype cvsub[NX * NP], ccsub[NX * NP];
    MC pMCsub[NP], xMCsub[NX], dxcvsub[NX], dxccsub[NX];
    UserData data;
    data = (UserData)user_data;


    /* Assign values to p, interval pI, McCormick pMC */
    for (int j = 0; j < NP; j++) {
        p[j] = data->p[j];
        pI[j] = I(pL[j], pU[j]);
        pMC[j] = MC(I(pL[j], pU[j]), p[j]);
        pMCsub[j] = MC(I(pL[j], pU[j]), p[j]);
        // DEbug: Cvode solver shows that convergence test failed repeatedly or with |h| = hmin
        // Solution: Missing initialization of parameters or state variables.
    }

    /* Initialize subgradeints*/
    // e.g. if NP = 2, cvsub_p1 = 1.0, 0.0, ccsub_p1 = 1.0, 0.0
    double sub[NP * NP] = { 0 };
    for (int j = 0; j < NP * NP; j++) {

        for (int i = 0; i < NP; i++) {

            if (j == i + i * NP) {
                sub[j] = 1.0;
            }
        }

    }

    /* Set subgradients for subpMC*/
    for (int j = 0; j < NP; j++) {

        pMCsub[j].sub(NP, &sub[j * NP], &sub[j * NP]);
    }


    /* Generate x_ori, xL, xU, xcv and xcc arrays using values from vector x*/
    for (int j = 0; j < NX; j++) {
        xori[j] = Ith(x, j + 1 + 0 * NX);
        xL[j] = Ith(x, j + 1 + 1 * NX);
        xU[j] = Ith(x, j + 1 + 2 * NX);
        xcv[j] = Ith(x, j + 1 + 3 * NX);
        xcc[j] = Ith(x, j + 1 + 4 * NX);
    }

    /* Generate cvsub, ccsub vectors using values from x*/
    for (int i = 0; i < NX * NP; i++) {
        
        cvsub[i] = Ith(x, 1 + 5 * NX + i );
        // e.g. for a system with NP=2, NX =2, cvsub should have the following structure:
        //// dx1^cv/dp1, dx2^cv/dp1, dx1^cv/dp2, dx2^cv/dp2;
        //update: dx1^cv/dp1, dx1^cv/dp2, dx2^cv/dp1, dx2^cv/dp2;

            // if there are three parameters, then the number of cvsub should be 2*NP
            // then the index of ccsub should start at i + 1 + (4+NP) * NY)

        ccsub[i] = Ith(x, 1 + 5 * NX + NX * NP + i);
        // e.g. for a system with NP=2, NX =2, cvsub should have the following structure:
        // dx1^cc/dp1, dx2^cc/dp1, dx1^cc/dp2, dx2^cc/dp2;
        //update: dx1^cc/dp1, dx1^cc/dp2, dx2^cc/dp1, dx2^cc/dp2;

    }



    /* Initialize interval xI, McCormick xMC */
    for (int j = 0; j < NX; j++) {
        xI[j] = I(xL[j], xU[j]);
        xMC[j] = MC(I(xL[j], xU[j]), xcv[j], xcc[j]);
        xMCsub[j] = MC(I(xL[j], xU[j]), xcv[j], xcc[j]);
    }


    /* Set subgradients for xMCusb with respect to p*/
    for (int j = 0; j < NX; j++)
    {
        xMCsub[j].sub(NP, &cvsub[j * NP], &ccsub[j * NP]);
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
        /* Subgradients computation*/

        /* Flattening the ith xMCsub (xiL,xiU,xicv,xicc,xicvsub,xiccsub)
        to (xiL,xiU,xicv,xicv,xicvsub,xicvsub)*/
        xMCsub[j] = MC(I(xL[j], xU[j]), xcv[j], xcv[j]);
        xMCsub[j].sub(NP, &cvsub[j * NP], &cvsub[j * NP]);
        /* Apply the flattened xMCsub into the original RHS function */
        dxcvsub[j] = Original_RHS(xMCsub, pMCsub, j);


        /* Flattening the ith xMCsub (xiL,xiU,xicv,xicc,xicvsub,xiccsub)
        to (xiL,xiU,xicv,xicv,xicvsub,xicvsub)*/
        xMCsub[j] = MC(I(xL[j], xU[j]), xcc[j], xcc[j]);
        xMCsub[j].sub(NP, &ccsub[j * NP], &ccsub[j * NP]);
        /* Apply the flattened xMCsub into the original RHS function */
        dxccsub[j] = Original_RHS(xMCsub, pMCsub, j);


        /* Unflattening the ith xMCsub*/
        xMCsub[j] = MC(I(xL[j], xU[j]), xcv[j], xcc[j]);
        xMCsub[j].sub(NP, &cvsub[j * NP], &ccsub[j * NP]);



        /*-------------------------------------------------------------*/
        /*-------------------------------------------------------------*/
        /* Construct dx vextor*/

        /* Construct x vector*/

        Ith(dx, j + 1 + 0 * NX) = xd[j];
        Ith(dx, j + 1 + 1 * NX) = dxL[j];
        Ith(dx, j + 1 + 2 * NX) = dxU[j];
        Ith(dx, j + 1 + 3 * NX) = dxcv[j];
        Ith(dx, j + 1 + 4 * NX) = dxcc[j];

        for (int i = 0; i < NP; i++) {
            Ith(dx, 1 + 5 * NX + j * NP + i) = dxcvsub[j].cvsub(i); //dxjcv/dpi

            Ith(dx, 1 + 5 * NX + NX * NP + j * NP + i) = dxccsub[j].ccsub(i); //dxjcc/dpi
        }

    }

    return(0);
}


/*
 * EwtSet function. Computes the error weights at the current solution.
 */

static int ewt(N_Vector y, N_Vector w, void* user_data)
{
    int i;
    realtype yy, ww, rtol, atol[NEQ];

    rtol = RTOL;

    for (int j = 0; j < NEQ; j++) {
        atol[j] = ATOL;
    }

    for (i = 1; i <= NEQ; i++) {
        yy = Ith(y, i);
        ww = rtol * SUNRabs(yy) + atol[i - 1];
        if (ww <= 0.0) return (-1);
        Ith(w, i) = 1.0 / ww;
    }

    return(0);
}

/*
 *--------------------------------------------------------------------
 * PRIVATE FUNCTIONS
 *--------------------------------------------------------------------
 */


 /*
  * Print current t, step count, order, stepsize, and solution.
  */


static void PrintOutput(void* cvode_mem, realtype t, N_Vector u)
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

    printf("                  Subgradient of Convex Relaxation");
    printf("%12.4e \n", udata[5 * NX + xi * NP + pi ]);

    printf("                  Subgradient of Concave Relaxation");
    printf("%12.4e \n", udata[5 * NX + NX * NP + xi * NP + pi]);

}


/*
* Return the results
*/
static void Getresults(N_Vector u)
{
    realtype* udata, hist[NEQ];

    /* capturing a returned array/pointer */
    udata = N_VGetArrayPointer(u);

    for (int i = 0; i < NEQ; i++) {

        hist[i] = udata[i];
    }

    int N = 0;

    realtype xi_hist = hist[N];
    realtype lb_hist = hist[N + NX];
    realtype ub_hist = hist[N + 2 * NX];
    realtype cv_hist = hist[N + 3 * NX];
    realtype cc_hist = hist[N + 4 * NX];


    ofstream outfile11;
    outfile11.open("/Users/yulanzhang/Desktop/Adjoint-paper/Example-updated/Example4/matlab_plots/x.txt", ios::app);
    outfile11 << xi_hist << "\n";
    outfile11.close();

    ofstream outfile21;
    outfile21.open("/Users/yulanzhang/Desktop/Adjoint-paper/Example-updated/Example4/matlab_plots/x_lb.txt", ios::app);
    outfile21 << lb_hist << "\n";
    outfile21.close();


    ofstream outfile31;
    outfile31.open("/Users/yulanzhang/Desktop/Adjoint-paper/Example-updated/Example4/matlab_plots/x_ub.txt", ios::app);
    outfile31 << ub_hist << "\n";
    outfile31.close();


    ofstream outfile41;
    outfile41.open("/Users/yulanzhang/Desktop/Adjoint-paper/Example-updated/Example4/matlab_plots/x_cv.txt", ios::app);
    outfile41 << cv_hist << "\n";
    outfile41.close();

    ofstream outfile52;
    outfile52.open("/Users/yulanzhang/Desktop/Adjoint-paper/Example-updated/Example4/matlab_plots/x_cc.txt", ios::app);
    outfile52 << cc_hist << "\n";
    outfile52.close();
}




/*
 * Print some final statistics from the CVODES memory.
 */

static void PrintFinalStats(void* cvode_mem)
{

    long int nst;
    long int nfe, nsetups, nni, ncfn, netf;
    long int nje, nfeLS;
    int retval;

    retval = CVodeGetNumSteps(cvode_mem, &nst);
    check_retval(&retval, "CVodeGetNumSteps", 1);
    retval = CVodeGetNumRhsEvals(cvode_mem, &nfe);
    check_retval(&retval, "CVodeGetNumRhsEvals", 1);
    retval = CVodeGetNumLinSolvSetups(cvode_mem, &nsetups);
    check_retval(&retval, "CVodeGetNumLinSolvSetups", 1);
    retval = CVodeGetNumErrTestFails(cvode_mem, &netf);
    check_retval(&retval, "CVodeGetNumErrTestFails", 1);
    retval = CVodeGetNumNonlinSolvIters(cvode_mem, &nni);
    check_retval(&retval, "CVodeGetNumNonlinSolvIters", 1);
    retval = CVodeGetNumNonlinSolvConvFails(cvode_mem, &ncfn);
    check_retval(&retval, "CVodeGetNumNonlinSolvConvFails", 1);


    /*
     * nfSe - number of calls to the sensitivity right-hand side function
     * nfes - number of calls to the user's ODE right-hand side function for
              the evaluation of sensitivity right-hand sides
     * netfS - number of error test failures
     * nsetupsS - number of calls to the linear solver setup function
     * nniS - number of nonlinear iterations performed
     * ncfnS - number of nonlinear convergence failures
     * nfeLS - number of calls made to the linear solver setup function
     * nje - the number of calls to the Jacobian function
    */

    retval = CVodeGetNumJacEvals(cvode_mem, &nje);
    check_retval(&retval, "CVodeGetNumJacEvals", 1);
    retval = CVodeGetNumLinRhsEvals(cvode_mem, &nfeLS);
    check_retval(&retval, "CVodeGetNumLinRhsEvals", 1);

    printf("\nFinal Statistics\n\n");
    printf("nst     = %5ld\n\n", nst);
    printf("nfe     = %5ld\n", nfe);
    printf("netf    = %5ld    nsetups  = %5ld\n", netf, nsetups);
    printf("nni     = %5ld    ncfn     = %5ld\n", nni, ncfn);

    printf("\n");
    printf("nje    = %5ld    nfeLS     = %5ld\n", nje, nfeLS);

}


static int check_retval(void* returnvalue, const char* funcname, int opt)
{
    int* retval;

    /* Check if SUNDIALS function returned NULL pointer - no memory allocated */
    if (opt == 0 && returnvalue == NULL) {
        fprintf(stderr,
            "\nSUNDIALS_ERROR: %s() failed - returned NULL pointer\n\n",
            funcname);
        return(1);
    }

    /* Check if retval < 0 */
    else if (opt == 1) {
        retval = (int*)returnvalue;
        if (*retval < 0) {
            fprintf(stderr,
                "\nSUNDIALS_ERROR: %s() failed with retval = %d\n\n",
                funcname, *retval);
            return(1);
        }
    }

    /* Check if function returned NULL pointer - no memory allocated */
    else if (opt == 2 && returnvalue == NULL) {
        fprintf(stderr,
            "\nMEMORY_ERROR: %s() failed - returned NULL pointer\n\n",
            funcname);
        return(1);
    }

    return(0);
}
