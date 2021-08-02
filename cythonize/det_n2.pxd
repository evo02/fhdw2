# cpdef double[:] det_p1(int pm, int lda, double[:, :] a, double[:] u, double[:] v, double[:] z, double s) nogil

# cdef void inv_p1(int pm, int lda, double[:, :] a, double det, double[:] v, double[:] w, double[:] z) nogil

# cdef double[:, :] det_m1(int pm, int lda, double[:, :] a, int r, int c) nogil

# cdef void inv_m1(int pm, int lda, double[:, :] a, int r, int c) nogil

# cdef double[:, :] det_r(int pm, int lda, double[:, :] a, int r, double[:] v) nogil

# cdef void inv_r(int pm, int lda, double[:,:] a, int r, double det, double[:] v, double[:] w, double[:] z) nogil

# cdef double full_inv(int pm, int lda, double[:, :] a) nogil

# cdef double det_p2(int pm, int lda, double[:, :] a, double[:, :] u2, double[:, :] v2, double[:, :] s2, double[:, :] c2) nogil