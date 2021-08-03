#!python
#cython: language_level=3, boundscheck=True

from scipy.linalg.cython_blas cimport dgemv, ddot, dger, dscal, daxpy, dswap, dcopy, dgemm
from scipy.linalg.cython_lapack cimport dgetrf, dgetri

import numpy as np
cimport numpy as np
cimport cython 

# ctypedef np.float64_t DOUBLE
# ctypedef np.uint32_t UINT32_t

# np.import_array()

cpdef double det_p1(int pm, int lda, double[:, :] a, double[:] u, double[:] v, double[:] z, double s) nogil: 
    """
    Det: add a row & a column

    integer :: lda, pm       ! leading dimension and actual size of A
    double  :: a(lda,lda)
    double  :: v(lda), u(lda), z(lda), s

    Input: 
        a(lda,lda), s, v(lda), u(lda)
        all arrays are of the size pm<lda 

    Output: 
        z(lda)
        det. ratio @ det_p1
    """
    cdef: 
        double alpha = 1.0
        double beta = 0.0
        int incx = 1
        int incy = 1
        double *a0 = &a[0,0]
    #z=a1 DOT u
    dgemv('N', &pm, &pm, &alpha, a0, &lda, &u[0], &incx, &beta, &z[0], &incy)
    
    #\lambda = v DOT z; \rho = s - \lambda; det = 1/\rh
    return s - ddot(&pm, &v[0], &incx, &z[0], &incy)

cpdef void inv_p1(int pm, int lda, double[:, :] a, double det, double[:] v, double[:] w, double[:] z) nogil:
    """
    Add row & column: update the inverse

    integer :: lda, pm       ! leading dimension and actual size of A
    double  :: a(lda,lda),det,v(lda),w(lda),z(lda) 

    Input: a(lda,lda), v(lda)
        z(lda), det --- as set by det_p1 
        all arrays are of the size pm<lda 

    Output: a(lda,lda) ==> (pm+1) * (pm+1) inverse
        pm = pm( @ input) + 1
    """
    cdef:
        double rho = 1/det
        double alpha = 1.0
        double beta = 0.0
        int incx = 1
        int incy = 1
        double *a0 = &a[0,0]
    #w = transp(a1)*v
    dgemv('T', &pm, &pm, &alpha, a0, &lda, &v[0], &incx, &beta, &w[0], &incy)

    #b^{-1} 
    #cdef void dger(int *m, int *n, d *alpha, d *x, int *incx, d *y, int *incy, d *a, int *lda) nogil:
    dger(&pm, &pm, &rho, &z[0], &incx, &w[0], &incy, a0, &lda)


    cdef:
        int size_n = a.shape[0]
        int size_m = a.shape[1]
        double *a1 = &a[0, size_m]
        double *a2 = &a[size_n, 0]
        double rho0 = -rho
    #last row
    #cdef void dscal(int *n, d *da, d *dx, int *incx) nogil:
    dscal(&pm, &beta, a1, &lda)

    #cdef void daxpy(int *n, d *da, d *dx, int *incx, d *dy, int *incy) nogil:
    daxpy(&pm, &rho0, &w[0], &incx, a1, &lda)

    #last column
    dscal(&pm, &beta, a2, &incx)
    daxpy(&pm, &rho0, &z[0], &incx, a2, &incy)

    #(pm+1,pm+1)
    a[pm+1, pm+1] = rho

    #update pm
    pm = pm + 1

cpdef double det_m1(int pm, int lda, double[:, :] a, int r, int c) nogil:
    """
    Det: drop r-th row & c-th column

    double function  det_m1(pm,lda,a,r,c)
    integer :: pm,lda       ! actual size & leading dimension of A
    double  :: a(lda,lda)    
	integer:: c,r
	integer :: s

    Input:  a(lda,lda), pm
    Output: det. ratio:  det BIG / det SMALL (=1/rho)
    """
    cdef:
        double s = 1.0
        double alpha = 1.0
    if c != pm: 
        s = -s
    
    if r!=pm:
        s = -s

    alpha *= ( s/a[c, r] )
    return alpha

cpdef void inv_m1(int pm, int lda, double[:, :] a, int r, int c) nogil:
    """
    Inv: drop r-th row and c-th column

    integer :: lda, pm       ! leading dimension and actual size of A
    double  :: a(lda,lda)
    integer :: c,r
	double :: rh

    Input: a(lda,lda) pm*pm
       r, c --- row and  column to be dropped

    Output: a(lda,lda) ==> (pm-1) * (pm-1)
        pm = pm( @ input ) - 1

    How:  swaps to-be-dropped and last row & cols, 
        and then drops the former ones

    The latter is done using: 
        a(pm,pm)         is   \rho
        a(1:pm,pm)       is  -\rho * z  -- last column
        a(pm,1:pm)       is  -\rho * w  -- last row
        a(1:pm-1,1:pm-1) is  a^{-1} + \rho z*w 
    """

    cdef:
        int incx = 1
        int incy = 1

        #may be need fortran/c contiguous
        #rows
        double *a0 = &a[c-1, 0]
        double *a1 = &a[0, pm-1]
        #columns 
        double *a2 = &a[r-1, pm-1]
        double *a3 = &a[pm-1, 0]
        double *a_full = &a[0, 0]

        #swap c-th and last row (c-th column of A <==> c-th row of A^{-1})
    if c!=pm: 
    #cdef void dswap(int *n, d *dx, int *incx, d *dy, int *incy) nogil:
        dswap(&pm, a0, &lda, a1, &lda)

        #swap r-th and last column
    if r!=pm:
    #cdef void dswap(int *n, d *dx, int *incx, d *dy, int *incy) nogil:
        dswap(&pm, a2, &incx, a3, &incy)

    #drop the last row & column
    cdef:
        double rh = -1.0/a[pm-1,pm-1]

    #update pm
    pm = pm - 1
    #stripe out the outer product z*w
    #cdef void dger(int *m, int *n, d *alpha, d *x, int *incx, d *y, int *incy, d *a, int *lda) nogil:
    dger(&pm, &pm, &rh, &a[0, pm], &incx, &a[pm,0], &lda, a_full, &lda)

cpdef double det_r(int pm, int lda, double[:, :] a, int r, double[:] v) nogil:
    """
    Det: change a single row
    
	integer :: lda, pm       ! leading dimension and actual size of A
	double  :: a(lda,lda), v(lda)
    integer :: r
    
    Input: 
        a(lda,lda) --- inverse matrix
        v(lda)     --- a row to be added
        r          --- a row number 
        all arrays are of the size pm<lda 

    Output:
        det. ratio @ det_r
    """
    #\lambda =  ( v DOT A_r )
    cdef:
        double alpha  = 1.0
        int incx = 1
        int incy = 1
        double *r0 = &a[0, r-1]
    #cdef d ddot(int *n, d *dx, int *incx, d *dy, int *incy) nogil:
    return alpha + ddot(&pm, &v[0], &incx, r0, &incy)

cpdef void inv_r(int pm, int lda, double[:,:] a, int r, double det, double[:] v, double[:] w, double[:] z) nogil:
    """
    inv: change a single row

    integer :: lda, pm       ! leading dimension and actual size of A
	double  :: a(lda,lda), v(lda),w(lda),z(lda),det
    integer :: r
    double  :: rho

    Input: 
        a(lda,lda)      --- inverse matrix
        v(lda)          --- a row to be added
        w(lda),z(lda)   --- working arrays
        r               --- row number 
        det             --- det ratio, as set by det_r
            all arrays are of the size pm<lda 

    Output:
        a(lda,lda) contains an updated inverse matrix
    """
    cdef:
        double rho = -1.0/det
        int incx = 1
        int incy = 1
        double alpha = 1.0
        double beta = 0.0
        double *a0 = &a[0,0]
        double *a_r = &a[0, r-1]

    #z_i = A_{i,r}
    #cdef void dcopy(int *n, d *dx, int *incx, d *dy, int *incy) nogil:
    dcopy(&pm, a_r, &incx, &z[0], &incy)

    #w = v DOT A 
    #cdef void dgemv(char *trans, int *m, int *n, d *alpha, d *a, int *lda, d *x, int *incx, d *beta, d *y, int *incy) nogil:
    dgemv('T', &pm, &pm, &alpha, a0, &lda, &v[0], &incx, &beta, &w[0], &incy)

    #A+ \rho* z DOT w^T
    #cdef void dger(int *m, int *n, d *alpha, d *x, int *incx, d *y, int *incy, d *a, int *lda) nogil:
    dger(&pm, &pm, &rho, &z[0], &incx, &w[0], &incy, a0, &lda)

    # one cannot get rid of z() array since if one otherwise plugs {a(1,r),1}
    # directly into dger(...) instead of {z,1}, it all goes nuts.
    # probably, dger() spoils the z array via blocking or the like.


cpdef double full_inv(int pm, int lda, double[:, :] a) nogil:
    """
    inv & det of a matrix (honest N**3)

    integer :: lda, pm  ! leading dimension and actual size of A
    double  :: a(lda,lda)

    integer, allocatable :: ipiv(:)
    double, allocatable :: work(:)
    integer :: info, i,lwork,icnt

    Input: 
        A(lda,lda), pm<lda 

    Output: 
        A contains the inverse
        full_inv contains the determinant

    """
    cdef:
        int info
        int lwork = lda
        #pm lenght
        int[:] ipiv
        #lwork lengt
        double[:] work
        double *a0 = &a[0,0]

	#allocate(ipiv(1:pm), work(1:lwork))
    dgetrf(&pm, &pm, a0, &lda, &ipiv[0], &info) 

    cdef:
        double res = 1.0
        int icnt = 0
    for i in range(pm):
        res *= a[i,i]
        if ipiv[i] != i:
            icnt+=1
        if icnt%2 == 1:
            res = -res
    dgetri(&pm, a0, &lda, &ipiv[0], &work[0], &lwork, &info)
    if info != 0:
        return -1

    return res

cpdef double det_p2(int pm, int lda, double[:, :] a, double[:, :] u, double[:, :] v, double[:, :] s, double[:, :] c) nogil:
    """
    Det: add a two rows & columns

    integer :: lda, pm       ! leading dimension and actual size of A
    real*8  :: a(lda,lda)
    real*8  :: v2(2,lda),u2(lda,2),s2(2,2),c2(lda,2)

    Input: 
        a(lda,lda), v2(2,lda), u2(lda,2), s2(2,2)
        all arrays are of the size pm<lda 

    Output:    
        det. ratio @ det_p2
    """
    cdef:
        int n = 2
        double alpha = 1.0
        double beta = 0.0
        double zalpha = -1.0
        double *a0 = &a[0, 0]
        double *u0 = &u[0, 0]
        double *v0 = &v[0, 0]
        double *s0 = &s[0, 0]
        double *c0 = &c[0, 0]
    #c2=a^{-1} DOT u2
    #cdef void dgemm(char *transa, char *transb, int *m, int *n, int *k, d *alpha, d *a, int *lda, d *b, int *ldb, d *beta, d *c, int *ldc) nogil:
    dgemm('N','N', &pm, &n, &pm, &alpha, a0, &lda, u0, &lda, &beta, c0, &lda)

    #\Rho = s2 - v2 DOT c2
    dgemm('N','N', &n, &n, &pm, &zalpha, v0, &n, c0, &lda, &alpha, s0, &n)

    #det = det \Rho [which is 2*2]
    return s[1,1]*s[2,2] - s[1,2]*s[2,1]