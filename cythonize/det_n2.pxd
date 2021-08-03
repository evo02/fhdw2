#cython: language_level=3, boundscheck=True

cpdef double det_p1(int pm, int lda, double[:, :] a, double[:] u, double[:] v, double[:] z, double s) nogil 

cpdef void inv_p1(int pm, int lda, double[:, :] a, double det, double[:] v, double[:] w, double[:] z) nogil

cpdef double det_m1(int pm, int lda, double[:, :] a, int r, int c) nogil

cpdef void inv_m1(int pm, int lda, double[:, :] a, int r, int c) nogil

cpdef double det_r(int pm, int lda, double[:, :] a, int r, double[:] v) nogil

cpdef void inv_r(int pm, int lda, double[:,:] a, int r, double det, double[:] v, double[:] w, double[:] z) nogil

cpdef double full_inv(int pm, int lda, double[:, :] a) nogil

cpdef double det_p2(int pm, int lda, double[:, :] a, double[:, :] u, double[:, :] v, double[:, :] s, double[:, :] c) nogil