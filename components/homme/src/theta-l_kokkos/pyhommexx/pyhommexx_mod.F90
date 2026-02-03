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
  logical, public :: output_to_screen = .true.

  ! Functions callable from C, mostly to detect whether some parts were already inited
  public :: init_parallel_f90
  public :: print_to_screen_f90
  public :: model_init_f90

contains

  subroutine init_parallel_f90 () bind(c)
    use iso_c_binding,  only: c_int
    use mpi,            only: MPI_COMM_WORLD
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

    integer :: ierr

    ! Initialize parallel structure
    print *, "init parallel, with comm:",MPI_COMM_WORLD
    par%comm = MPI_COMM_WORLD
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
    call reset_cxx_comm (INT(MPI_COMM_WORLD,kind=c_int))

    npes = par%nprocs
    iam  = par%rank
    masterproc = par%masterproc
  end subroutine init_parallel_f90

  subroutine print_to_screen_f90(enabled) bind(c)
    use iso_c_binding, only: c_bool
    use iso_fortran_env, only: output_unit, error_unit

    ! Inputs
    logical(kind=c_bool), intent(in), value :: enabled

    if (enabled) then
      print *, "enable output requested..."
      if (.not. output_to_screen) then
        print *, "must enable output..."
        open(unit=output_unit,file='/dev/tty',status='replace')
        open(unit=error_unit,file='/dev/tty',status='replace')
        print *, "output disabled"
      else
        print *, "output was already enabled..."
      endif
    else
      print *, "disable output requested..."
      if (output_to_screen) then
        print *, "must disable output..."
        open(unit=output_unit,file='/dev/null',status='replace')
        open(unit=error_unit,file='/dev/null',status='replace')
      else
        print *, "output was already disabled..."
      endif
    endif
  end subroutine print_to_screen_f90

  subroutine get_num_unique_pts_f90 (num_ptr) bind(c)
    use iso_c_binding,  only: c_ptr, c_f_pointer, c_int
    use dimensions_mod, only: nelemd

    ! Inputs
    type (c_ptr), intent(in) :: num_ptr

    integer(kind=c_int), pointer :: num(:)
    integer :: ie

    call c_f_pointer(num_ptr,num,[nelemd])

    do ie=1,nelemd
      num(ie) = elem(ie)%idxP%NumUniquePts
    enddo

  end subroutine get_num_unique_pts_f90

  subroutine get_unique_pts_f90 (ia_ptr,ja_ptr) bind(c)
    use iso_c_binding,  only: c_ptr, c_f_pointer, c_int
    use dimensions_mod, only: nelemd, np

    ! Inputs
    type (c_ptr), intent(in) :: ia_ptr,ja_ptr

    integer(kind=c_int), pointer :: ia(:,:)
    integer(kind=c_int), pointer :: ja(:,:)

    integer :: ie, n

    call c_f_pointer(ia_ptr,ia,[np*np,nelemd])
    call c_f_pointer(ja_ptr,ja,[np*np,nelemd])

    do ie=1,nelemd
      do n=1,elem(ie)%idxP%NumUniquePts
        ja(n,ie) = elem(ie)%idxP%ia(n)-1 ! These are for C++, so use 0-based indexing
        ia(n,ie) = elem(ie)%idxP%ja(n)-1
      enddo
    enddo

  end subroutine get_unique_pts_f90

  subroutine read_params_f90 (nl_fname_c) bind(c)
    use iso_c_binding,   only: c_ptr, c_f_pointer, C_NULL_CHAR
    use namelist_mod,    only: readnl
    use control_mod,     only: MAX_STRING_LEN
    use prim_driver_mod, only: prim_init_simulation_params

    ! Inputs
    type (c_ptr), intent(in) :: nl_fname_c

    ! Locals
    integer :: str_len
    character(len=MAX_STRING_LEN), pointer :: nl_fname_ptr

    ! Force namelist mod to read from file
    call c_f_pointer(nl_fname_c,nl_fname_ptr)
    str_len = index(nl_fname_ptr, C_NULL_CHAR) - 1
    call readnl(par,trim(nl_fname_ptr(1:str_len)))

    call prim_init_simulation_params()
  end subroutine read_params_f90

  subroutine model_init_f90 () bind(c)
    use iso_c_binding,   only: c_ptr, c_f_pointer, C_NULL_CHAR
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
  end subroutine model_init_f90

  subroutine prim_finalize_f90 () bind(c)
    use prim_driver_mod, only: prim_finalize

    call prim_finalize()
  end subroutine prim_finalize_f90

end module pyhommexx_mod
