! A matrix multiplication program written to practice using OpenACC directives.
! Run with command line arguments ./a.out n m p
! where n m p define the dimensions of an n x m matrix A, an m x p matrix B,
! and an n x p matrix C.
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
! Check if command line args exist
IF(COMMAND_ARGUMENT_COUNT() /= 3) THEN
  WRITE(*, *) "Usage: ./matr_mult n m p"
  WRITE(*, *) "For an n x m matrix A, an m x p matrix B, and an n x p matrix C"
  STOP
END IF
! Read command line args
CALL GET_COMMAND_ARGUMENT(1, nch)
CALL GET_COMMAND_ARGUMENT(2, mch)
CALL GET_COMMAND_ARGUMENT(3, pch)
READ(nch, *) n
READ(mch, *) m
READ(pch, *) p
! Allocate arrays and initialize them.
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

! Copy data to device and use parallel loops. In my test case I used the args
! 1200 1400 1600. The tile directive was faster than collapse, and tile(8,8)
! was slightly faster than the other combinations. vector_length(128) was
! slightly faster than other lengths. On Leia, my test case ran in 1.05s
! using the -fast flag without managed memory. It ran in 0.03s with the -fast
! flag and managed memory.
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

! Compare my implementation on the device to the intrinsic MATMUL calculated
! on the host. Check for correctness within a tolerance. Print success or
! failure and report the wall time for computing on the device.
Ch = MATMUL(Ah, Bh)
IF (ALL(ABS(Ch - Cd) < TOL)) THEN
  WRITE(*, *) 'Matrix multiplication calculated successfully.'
ELSE
  WRITE(*, *) 'Matrix multiplcation FAILED. Device calculation does not equal host.'
END IF
WRITE(*, '(A,F10.6,A)') 'Wall time to calculate on device: ', acc_end - acc_st, 's'
END PROGRAM matr_mult 
