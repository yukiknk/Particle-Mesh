C     PM: send the MPI_ALLTOALL(V) inside FFTE in single precision
C     (COMPLEX*8) when requested.
C
C     PMA2AV / PMA2A replace MPI_ALLTOALLV / MPI_ALLTOALL. Call them
C     from inside an OpenMP parallel region by all threads (the caller
C     needs no BARRIER / MASTER). IPMCF in /PMCFLT/ (set from C++):
C       0 : send as MPI_DOUBLE_COMPLEX as before (default)
C       1 : round to COMPLEX*8 just before sending and widen to
C           COMPLEX*16 just after receiving
C     FFT and reordering stay in double precision. The rounding error
C     is about 6E-8 (relative) per transfer. PMTCOM counts only the MPI
C     call (the conversion time is not included).
C     IPMCF must be the same on all ranks.
      SUBROUTINE PMA2AV(A,ISC,ISD,B,IRC,IRD,NPU,ICOMM)
      IMPLICIT NONE
      INCLUDE 'mpif.h'
      INTEGER NPU,ICOMM
      INTEGER ISC(*),ISD(*),IRC(*),IRD(*)
      COMPLEX*16 A(*),B(*)
      DOUBLE PRECISION PMTCOM,PMT0
      INTEGER IPMCF
      COMMON /PMFTIM/ PMTCOM,PMT0
      COMMON /PMCFLT/ IPMCF
      COMPLEX*8, ALLOCATABLE, SAVE :: SBUF(:),RBUF(:)
      INTEGER*8, SAVE :: NSA=0,NRA=0
      INTEGER*8 NS,NR
      INTEGER I,IERR,IST1,IST2
C
      IF (IPMCF .EQ. 0) THEN
!$OMP BARRIER
!$OMP MASTER
        PMT0=MPI_WTIME()
        CALL MPI_ALLTOALLV(A,ISC,ISD,MPI_DOUBLE_COMPLEX,B,IRC,IRD,
     1                     MPI_DOUBLE_COMPLEX,ICOMM,IERR)
        PMTCOM=PMTCOM+(MPI_WTIME()-PMT0)
!$OMP END MASTER
!$OMP BARRIER
        RETURN
      END IF
C
      NS=0
      NR=0
      DO 10 I=1,NPU
        IF (ISC(I) .GT. 0) NS=MAX(NS,INT(ISD(I),8)+ISC(I))
        IF (IRC(I) .GT. 0) NR=MAX(NR,INT(IRD(I),8)+IRC(I))
   10 CONTINUE
!$OMP BARRIER
!$OMP SINGLE
      IF (NSA .LT. NS) THEN
        IF (ALLOCATED(SBUF)) DEALLOCATE(SBUF)
        ALLOCATE(SBUF(NS),STAT=IST1)
        NSA=NS
      ELSE
        IST1=0
      END IF
      IF (NRA .LT. NR) THEN
        IF (ALLOCATED(RBUF)) DEALLOCATE(RBUF)
        ALLOCATE(RBUF(NR),STAT=IST2)
        NRA=NR
      ELSE
        IST2=0
      END IF
      IF (IST1 .NE. 0 .OR. IST2 .NE. 0) THEN
        WRITE(*,*) 'PMA2AV: cannot allocate single precision buffer'
        CALL MPI_ABORT(MPI_COMM_WORLD,1,IERR)
      END IF
!$OMP END SINGLE
      DO 20 I=1,NPU
        IF (ISC(I) .GT. 0) CALL PMTOF(A(ISD(I)+1),SBUF(ISD(I)+1),
     1                                INT(ISC(I),8))
   20 CONTINUE
!$OMP BARRIER
!$OMP MASTER
      PMT0=MPI_WTIME()
      CALL MPI_ALLTOALLV(SBUF,ISC,ISD,MPI_COMPLEX,RBUF,IRC,IRD,
     1                   MPI_COMPLEX,ICOMM,IERR)
      PMTCOM=PMTCOM+(MPI_WTIME()-PMT0)
!$OMP END MASTER
!$OMP BARRIER
      DO 30 I=1,NPU
        IF (IRC(I) .GT. 0) CALL PMTOD(RBUF(IRD(I)+1),B(IRD(I)+1),
     1                                INT(IRC(I),8))
   30 CONTINUE
!$OMP BARRIER
      RETURN
      END
C
C     MPI_ALLTOALL version. N: elements per peer, NPU: size of ICOMM.
      SUBROUTINE PMA2A(A,B,N,NPU,ICOMM)
      IMPLICIT NONE
      INCLUDE 'mpif.h'
      INTEGER N,NPU,ICOMM
      COMPLEX*16 A(*),B(*)
      DOUBLE PRECISION PMTCOM,PMT0
      INTEGER IPMCF
      COMMON /PMFTIM/ PMTCOM,PMT0
      COMMON /PMCFLT/ IPMCF
      COMPLEX*8, ALLOCATABLE, SAVE :: SBUF(:),RBUF(:)
      INTEGER*8, SAVE :: NSA=0
      INTEGER*8 NT
      INTEGER IERR,IST1,IST2
C
      IF (IPMCF .EQ. 0) THEN
!$OMP BARRIER
!$OMP MASTER
        PMT0=MPI_WTIME()
        CALL MPI_ALLTOALL(A,N,MPI_DOUBLE_COMPLEX,B,N,
     1                    MPI_DOUBLE_COMPLEX,ICOMM,IERR)
        PMTCOM=PMTCOM+(MPI_WTIME()-PMT0)
!$OMP END MASTER
!$OMP BARRIER
        RETURN
      END IF
C
      NT=INT(N,8)*NPU
!$OMP BARRIER
!$OMP SINGLE
      IF (NSA .LT. NT) THEN
        IF (ALLOCATED(SBUF)) DEALLOCATE(SBUF)
        IF (ALLOCATED(RBUF)) DEALLOCATE(RBUF)
        ALLOCATE(SBUF(NT),STAT=IST1)
        ALLOCATE(RBUF(NT),STAT=IST2)
        NSA=NT
        IF (IST1 .NE. 0 .OR. IST2 .NE. 0) THEN
          WRITE(*,*) 'PMA2A: cannot allocate single precision buffer'
          CALL MPI_ABORT(MPI_COMM_WORLD,1,IERR)
        END IF
      END IF
!$OMP END SINGLE
      CALL PMTOF(A,SBUF,NT)
!$OMP BARRIER
!$OMP MASTER
      PMT0=MPI_WTIME()
      CALL MPI_ALLTOALL(SBUF,N,MPI_COMPLEX,RBUF,N,
     1                  MPI_COMPLEX,ICOMM,IERR)
      PMTCOM=PMTCOM+(MPI_WTIME()-PMT0)
!$OMP END MASTER
!$OMP BARRIER
      CALL PMTOD(RBUF,B,NT)
!$OMP BARRIER
      RETURN
      END
C
C     Convert a shared array, split among all threads (NOWAIT: the
C     caller places the BARRIER).
      SUBROUTINE PMTOF(Z,C,N)
      IMPLICIT NONE
      INTEGER*8 N,I
      COMPLEX*16 Z(*)
      COMPLEX*8 C(*)
!$OMP DO
      DO 10 I=1,N
        C(I)=CMPLX(SNGL(DBLE(Z(I))),SNGL(DIMAG(Z(I))))
   10 CONTINUE
!$OMP END DO NOWAIT
      RETURN
      END
C
      SUBROUTINE PMTOD(C,Z,N)
      IMPLICIT NONE
      INTEGER*8 N,I
      COMPLEX*16 Z(*)
      COMPLEX*8 C(*)
!$OMP DO
      DO 10 I=1,N
        Z(I)=DCMPLX(DBLE(REAL(C(I))),DBLE(AIMAG(C(I))))
   10 CONTINUE
!$OMP END DO NOWAIT
      RETURN
      END
