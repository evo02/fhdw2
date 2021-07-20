from scipy.linalg.cython_blas import dgemv, ddot, dger, dscal, daxpy, dswap, dcopy, dgetrf, dgetri, dgemm
from cython.view cimport array

cpdef double[:] det_p1(int pm, int lda, double[:, :] a, double[:] u, double[:] v, double[:] z, double s) nogil: 
"""
Det: add a row & a column
integer :: lda, pm       ! leading dimension and actual size of A
double  :: a(lda,lda)
double  :: v(lda), u(lda), z(lda), s

Input: a(lda,lda), s, v(lda), u(lda)
        all arrays are of the size pm<lda 
Output: z(lda)
        det. ratio @ det_p1
"""
# cdef void dgemv(char *trans, int *m, int *n, d *alpha, d *a, int *lda, d *x, int *incx, d *beta, d *y, int *incy) nogil:
#      _fortran_dgemv(trans, m, n, alpha, a, lda, x, incx, beta, y, incy) 

    #z=a1 DOT u
    dgemv('N', pm, pm, 1.d0, a, lda, u, 1, 0.d0, z, 1)

    #\lambda = v DOT z; \rho = s - \lambda; det = 1/\rh
    return s - ddot(pm,v,1,z,1)

cdef void inv_p1(int pm, int lda, double[:, :] a, double det, double[:] v, double[:] w, double[:] z) nogil:
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
    cdef double rho = 1/det
    #w = transp(a1)*v
    dgemv('T', pm, pm, 1.d0, a, lda, v, 1, 0.d0, w, 1)

    #b^{-1} 
    dger(pm, pm, rho, z, 1, w, 1, a, lda)

    #last row
    dscal(pm,0.d0,a(pm+1,1),lda) ! set =0 first
    daxpy(pm, -rho, w,1,a(pm+1,1),lda)

    #last column
    call dscal(pm,0.d0,a(1,pm+1),1) ! set =0 first
    call daxpy(pm, -rho, z,1,a(1,pm+1),1)

    #(pm+1,pm+1)
    a(pm+1,pm+1)=rho

    #update pm
    pm=pm+1

cdef double[:, :] det_m1(int pm, int lda, double[:, :] a, int r, int c) nogil:
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
    cdef int s=1
    if(c/=pm)s=-s; if(r/=pm)s=-s
	return 1.d0 * s/a(c,r)

cdef void inv_m1(int pm, int lda, double[:, :] a, int r, int c) nogil:
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
    #swap c-th and last row (c-th column of A <==> c-th row of A^{-1})
    if(c/=pm): dswap(pm, a(c,1), lda, a(pm,1), lda)

    #swap r-th and last column
    if(r/=pm) call dswap(pm,a(1,r),1,a(1,pm),1)

    #drop the last row & column
    cdef double rh = -1.d0/a(pm,pm)

    #stripe out the outer product z*w
    dger(pm-1, pm-1, rh, a(1,pm), 1, a(pm,1), lda, a, lda)

    #update pm
    pm = pm - 1

cdef double[:, :] det_r(int pm, int lda, double[:, :] a, int r, double[:] v) nogil:
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
    return 1.d0 + ddot(pm,v,1,a(1,r),1)

cdef void inv_r(int pm, int lda, double[:,:] a, int r, double det, double[:] v, double[:] w, double[:] z) nogil:
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
    cdef double rho = -1.d0/det
    #z_i = A_{i,r}
    dcopy(pm,a(1,r),1,z,1)

    #w = v DOT A 
    dgemv('T',pm,pm,1.d0,a,lda,v,1,0.d0,w,1)

    #A+ \rho* z DOT w^T
    dger(pm,pm,rho,z,1,w,1,a,lda)

    # one cannot get rid of z() array since if one otherwise plugs {a(1,r),1}
    # directly into dger(...) instead of {z,1}, it all goes nuts.
    # probably, dger() spoils the z array via blocking or the like.


cdef double full_inv(int pm, int lda, double[:, :] a) nogil:
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
    cdef int lwork = lda
    cdef int[:] ipiv, work
	allocate(ipiv(1:pm), work(1:lwork))

	dgetrf(pm,pm,a,lda,ipiv, info) 

    full_inv = 1d0
        icnt=0
        do i=1,pm
        full_inv = full_inv * a(i,i)
           if (ipiv(i).ne.i) then
             icnt = icnt+1
           endif
        enddo
        if (mod(icnt,2).eq.1) full_inv = -full_inv



	dgetri(pm,a,lda,ipiv,work,lwork,info )
	if(info/=0) print*,'dgetri info = ',info
	deallocate( work,ipiv )
    return -1 #!!

cdef double det_p2(int pm, int lda, double[:, :] a, double[:, :] u2, double[:, :] v2, double[:, :] s2, double[:, :] c2) nogil:
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
    #c2=a^{-1} DOT u2
    dgemm('N','N',pm,2,pm,1.d0,a,lda,u2,lda,0.d0,c2,lda)

    #\Rho = s2 - v2 DOT c2
    dgemm('N','N',2,2,pm,-1.d0,v2,2,c2,lda,1.d0,s2,2)

    #det = det \Rho [which is 2*2]
    return s2(1,1)*s2(2,2) - s2(1,2)*s2(2,1)