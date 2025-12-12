module pyhommexx_mod

  use domain_mod,          only: domain1d_t
  use derivative_mod_base, only: derivative_t
  use element_mod,         only: element_t
  use parallel_mod,        only: parallel_t, abortmp
  use time_mod,            only: timelevel_t
  use hybvcoord_mod,       only: hvcoord_t
  use hybrid_mod,          only: hybrid_t
  use prim_driver_base,    only: deriv => deriv1

  implicit none
  private

  ! Making deriv (which is an alias to deriv1 in prim_driver_base) available
  public :: deriv

  type (element_t),    pointer :: elem(:) => null()
  type (domain1d_t),   pointer :: dom_mt(:) => null()
  type (parallel_t)            :: par
  type (timelevel_t)           :: tl
  type (hybrid_t)              :: hybrid
  type (hvcoord_t)             :: hvcoord

  integer, public :: npes        = 1
  integer, public :: iam         = 0
  logical, public :: masterproc  = .false.

  ! Functions callable from C, mostly to detect whether some parts were already inited
  public :: init_parallel_f90
  public :: prim_init_f90

contains

  subroutine init_parallel_f90 (f_comm) bind(c)
    use iso_c_binding,  only: c_int
    use parallel_mod,   only: initmp_from_par, abortmp
    use dimensions_mod, only: npart
    use hybrid_mod,     only: hybrid_create
    interface
      subroutine reset_cxx_comm (f_comm) bind(c)
        use iso_c_binding, only: c_int
        !
        ! Inputs
        !
        integer(kind=c_int), intent(in) :: f_comm
      end subroutine reset_cxx_comm
    end interface

    ! Input(s)
    integer (kind=c_int), intent(in) :: f_comm

    ! Local(s)
    integer :: ierr

    ! Initialize parallel structure
    par%comm = f_comm
    par%root = 0
    par%dynproc = .true.

    call MPI_comm_rank(par%comm,par%rank,ierr)
    call MPI_comm_size(par%comm,par%nprocs,ierr)

    par%masterproc = .false.
    if (par%rank .eq. par%root) par%masterproc = .true.

    call initmp_from_par(par)

    ! No horizontal threading in f90 when using C++
    hybrid = hybrid_create(par,0,1)

    ! Set number of mesh partitions equal to the number of ranks
    npart = par%nprocs

    ! Init the comm in Hommexx
    call reset_cxx_comm (f_comm)

    npes = par%nprocs
    iam  = par%rank
    masterproc = par%masterproc
  end subroutine init_parallel_f90

  subroutine prim_init_f90 () bind(c)
    use prim_driver_mod, only: prim_init1, prim_init2
    use dimensions_mod,  only: nelemd
    use control_mod,     only: vfile_mid, vfile_int
    use parallel_mod,    only: abortmp
    use hybvcoord_mod,   only: hvcoord_init

    ! Locals
    integer :: ierr

    call prim_init1(elem,  par,dom_mt,tl)

    ! Initialize the vertical coordinate
    hvcoord = hvcoord_init(vfile_mid, vfile_int, .true., hybrid%masterthread, ierr)
    if (ierr /= 0) then
       call abortmp("error in hvcoord_init")
    end if

    call prim_init2(elem, hybrid,1,nelemd,tl, hvcoord)
  end subroutine prim_init_f90

  subroutine prim_forward_f90 () bind(c)
    use time_mod,        only: tstep
    use prim_driver_mod, only: prim_run_subcycle
    use dimensions_mod,  only: nelemd

    call prim_run_subcycle(elem, hybrid,1,nelemd, tstep, .false., tl, hvcoord,1)
  end subroutine prim_forward_f90

  subroutine prim_finalize_f90 () bind(c)
    use prim_driver_mod, only: prim_finalize

    call prim_finalize()
  end subroutine prim_finalize_f90

end module pyhommexx_mod
