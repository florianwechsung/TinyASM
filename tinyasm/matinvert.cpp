#include <petsc.h>
//#include <mkl_lapack.h>
//#include <mkl.h>
#include <petscblaslapack.h>

#include <numeric>
#include <vector>
using namespace std;

PetscErrorCode mymatinvert(PetscInt* n, PetscScalar* mat, PetscInt* piv, PetscInt* info, PetscScalar* work) {
    //PetscCallBLAS("LAPACKgetrf",LAPACKgetrf_(&dof,&dof,&matValuesPerBlock[p][0],&dof,&piv[0],&info));
    //PetscCallBLAS("LAPACKgetri",LAPACKgetri_(&dof,&matValuesPerBlock[p][0], &dof, &piv[0],&fwork[0],&dof,&info));
	//dgetrf(n,n,mat,n,piv,info);
	//dgetri(n,mat, n, piv,work,n,info);

	//LAPACKE_mkl_dgetrfnpi(LAPACK_COL_MAJOR,*n,*n,*n,mat,*n);
	//LAPACKE_dgetrf(LAPACK_COL_MAJOR,*n,*n,mat,*n, piv);
    //LAPACKE_dgetri(LAPACK_COL_MAJOR, *n,mat, *n, piv);
    PetscCallBLAS("LAPACKgetrf",LAPACKgetrf_(n,n,mat,n,piv,info));
    PetscCallBLAS("LAPACKgetri",LAPACKgetri_(n,mat, n, piv,work,n,info));
	return 0;
}

