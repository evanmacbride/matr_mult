PROGRAM matr_mult
USE omp_lib
IMPLICIT NONE
INTEGER, PARAMETER :: DBL = SELECTED_REAL_KIND(p = 12)    ! Double precision
INTEGER :: i, j, k                                        ! Index variables
INTEGER :: n, m, p                                        ! Matrix dimensions
CHARACTER(len=12) :: nch, mch, pch                        ! Command line args
REAL(KIND=DBL) :: r                                       ! A random number
REAL(KIND=DBL) :: loop_sum                                ! The loop sum
REAL(KIND=DBL), DIMENSION(:,:), ALLOCATABLE :: Ad, Bd, Cd ! Compute on device
REAL(KIND=DBL), DIMENSION(:,:), ALLOCATABLE :: Ah, Bh, Ch ! Compute on host
REAL(KIND=DBL), PARAMETER :: TOL = 1.E-9_DBL             ! Error-checking tol
REAL(KIND=DBL) :: acc_st, acc_end                         ! Timing vars
IF(COMMAND_ARGUMENT_COUNT() /= 3) THEN
  WRITE(*, *) "Usage: ./matr_mult n m p"
  WRITE(*, *) "For an n x m matrix A, an m x p matrix B, and an n x p matrix C"
  STOP
END IF

CALL GET_COMMAND_ARGUMENT(1, nch)
CALL GET_COMMAND_ARGUMENT(2, mch)
CALL GET_COMMAND_ARGUMENT(3, pch)
READ(nch, *) n
READ(mch, *) m
READ(pch, *) p

ALLOCATE(Ad(n,m),Bd(m,p),Cd(n,p))
DO i = 1, n
  DO j = 1, m
    CALL RANDOM_NUMBER(r)
    Ad(i,j) = r
  END DO
END DO
DO i = 1, m
  DO j = 1, p
    CALL RANDOM_NUMBER(r)
    Bd(i,j) = r
  END DO
END DO
Cd = 0._DBL
Ah = Ad
Bh = Bd
Ch = Cd

acc_st = OMP_GET_WTIME()
!$acc data copyin(Ad, Bd) copyout(Cd)
!$acc parallel loop tile(8,8) private(loop_sum) vector_length(128)
DO i = 1, n
  DO j = 1, p
    loop_sum = 0._DBL
    !$acc loop reduction(+:loop_sum)
    DO k = 1, m
      loop_sum = loop_sum + Ad(i,k) * Bd(k,j)
    END DO
    Cd(i,j) = loop_sum
  END DO
END DO
!$acc end data
acc_end = OMP_GET_WTIME()
Ch = MATMUL(Ah, Bh)

IF (ALL(ABS(Ch - Cd) < TOL)) THEN
  WRITE(*, *) 'Matrix multiplication calculated successfully.'
ELSE
  WRITE(*, *) 'Matrix multiplcation FAILED. Device calculation does not equal host.'
END IF
WRITE(*, *) 'Wall time to calculate on device: ', acc_end - acc_st
END PROGRAM matr_mult 
