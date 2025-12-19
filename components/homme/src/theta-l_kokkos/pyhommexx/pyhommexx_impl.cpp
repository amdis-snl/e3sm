#include "pyhommexx_decl.hpp"
#include "pyhommexx_utils.hpp"
#include "pyhommexx_c2f.hpp"

#include "Context.hpp"
#include "ElementsState.hpp"
#include "ElementsGeometry.hpp"
#include "SimulationParams.hpp"
#include "TimeLevel.hpp"
#include "Types.hpp"
#include "mpi/Comm.hpp"
#include "utilities/ViewUtils.hpp"

#include <nanobind/ndarray.h>

namespace Homme {
extern "C" {
void prim_run_subcycle_c (const Real& dt, int& nstep, int& nm1, int& n0, int& np1,
                          const int& next_output_step, const int& nsplit_iteration);
}

namespace pyhommexx {

double* vp2dp (void* p)
{
  return reinterpret_cast<double*>(p);
}
const double* vp2cdp (void* p)
{
  return reinterpret_cast<const double*>(p);
}

template<typename T, int N1, int N2, int N3, int N4>
ExecViewUnmanaged<T*[N2][N3][N4]> sv_1(const ExecViewUnmanaged<T*[N1][N2][N3][N4]>& v, int k)
{
  constexpr auto ALL = Kokkos::ALL();
  return Kokkos::subview(v,ALL,k,ALL,ALL,ALL);
}

template<typename T, int N1, int N2, int N3, int N4, int N5>
ExecViewUnmanaged<T*[N2][N3][N4][N5]> sv_1(const ExecViewUnmanaged<T*[N1][N2][N3][N4][N5]>& v, int k)
{
  constexpr auto ALL = Kokkos::ALL();
  return Kokkos::subview(v,ALL,k,ALL,ALL,ALL,ALL);
}

template<typename NBArrayT>
void check_shape(const NBArrayT& arr, const std::vector<int>& shape)
{
  assert (arr.ndim()==shape.size());
  for (size_t i=0; i<shape.size(); ++i) {
    assert (static_cast<int>(arr.shape(i))==shape[i]);
  }
}

void init (nb::object py_comm, const nb::str& nml_filename)
{
  const auto& fcomm = get_f_comm(get_c_comm(py_comm));
  init_parallel_f90(fcomm);

  auto nml_filename_c = nml_filename.c_str();
  prim_init_f90(nml_filename_c);
}

nb::dict get_params()
{
  const auto& c = Context::singleton();
  const auto& s = c.get<SimulationParams>();
  const auto& e = c.get<ElementsState>();

  nb::dict params;

  params["nelemd"] = e.num_elems();
  params["rsplit"] = s.rsplit;
  params["qsplit"] = s.qsplit;
  params["qsize"] = s.qsize;
  params["qsize_d"] = QSIZE_D;
  params["np"] = NP;
  params["nlev"] = NUM_PHYSICAL_LEV;
  params["hydrostatic"] = s.theta_hydrostatic_mode;
  params["nu"] = s.nu;

  int nelem = e.num_elems();
  MPI_Allreduce(MPI_IN_PLACE, &nelem, 1, MPI_INT, MPI_SUM, c.get<Comm>().mpi_comm());
  int ne = static_cast<int>(std::sqrt(nelem / 6));
  params["ne"] = ne;

  return params;
}

void get_num_unique_pts (nb::ndarray<int>& n)
{
  const auto& c = Context::singleton();
  const auto& state = c.get<ElementsState> ();
  int num_elems = state.num_elems();
  assert(n.ndim()==1);
  assert(n.shape(0)==num_elems);
  int* data = n.data();
  get_num_unique_pts_f90(data);
}
void get_unique_pts (nb::ndarray<int>& ia,
                     nb::ndarray<int>& ja)
{
  const auto& c = Context::singleton();
  const auto& state = c.get<ElementsState> ();
  int num_elems = state.num_elems();
  assert(ia.ndim()==2);
  assert(ia.shape(0)==num_elems);
  assert(ia.shape(1)==NP*NP);
  assert(ja.ndim()==2);
  assert(ja.shape(0)==num_elems);
  assert(ja.shape(1)==NP*NP);

  int* ia_ptr = ia.data();
  int* ja_ptr = ja.data();
  get_unique_pts_f90(ia_ptr,ja_ptr);

}

void set_state (const nb::ndarray<double>& uv,
                const nb::ndarray<double>& vthdp,
                const nb::ndarray<double>& dp)
{
  const auto& c = Context::singleton();
  const auto& state = c.get<ElementsState> ();
  const auto& tl = c.get<TimeLevel> ();

  const int nelem = state.num_elems();
  const int n0 = tl.n0;

  std::vector<int> vector3d_shape = {nelem,2,NP,NP,NUM_PHYSICAL_LEV};
  std::vector<int> scalar3d_shape = {nelem,  NP,NP,NUM_PHYSICAL_LEV};

  check_shape(uv,vector3d_shape);
  check_shape(vthdp,scalar3d_shape);
  check_shape(dp,scalar3d_shape);
  assert ((int)uv.dtype().bits==64);
  assert ((int)vthdp.dtype().bits==64);
  assert ((int)dp.dtype().bits==64);

  ExecViewUnmanaged<const double*****> uv_k (vp2cdp(uv.data()),nelem,2,NP,NP,NUM_PHYSICAL_LEV);
  ExecViewUnmanaged<const double****>  vth_k(vp2cdp(vthdp.data()),nelem,NP,NP,NUM_PHYSICAL_LEV);
  ExecViewUnmanaged<const double****>  dp_k (vp2cdp(dp.data()),nelem,NP,NP,NUM_PHYSICAL_LEV);

  auto v_s = state.m_v;
  auto vtheta_dp_s = state.m_vtheta_dp;
  auto dp3d_s = state.m_dp3d;

  using policy_t = Kokkos::MDRangePolicy<ExecSpace,Kokkos::Rank<4>>;
  policy_t p ({0,0,0,0},{nelem,NP,NP,NUM_PHYSICAL_LEV});
  auto copy = KOKKOS_LAMBDA (int ie, int ip, int jp, int k) {
    int lev = k / VECTOR_SIZE;
    int vec = k % VECTOR_SIZE;
    v_s(ie,n0,0,ip,jp,lev)[vec] = uv_k(ie,0,ip,jp,k);
    v_s(ie,n0,1,ip,jp,lev)[vec] = uv_k(ie,1,ip,jp,k);
    vtheta_dp_s(ie,n0,ip,jp,lev)[vec] = vth_k(ie,ip,jp,k);
    dp3d_s(ie,n0,ip,jp,lev)[vec] = dp_k(ie,ip,jp,k);
  };
  Kokkos::parallel_for(p,copy);
}

void get_state (nb::ndarray<double>& uv,
                nb::ndarray<double>& vthdp,
                nb::ndarray<double>& dp)
{
  const auto& c = Context::singleton();
  const auto& state = c.get<ElementsState> ();
  const auto& tl = c.get<TimeLevel> ();

  const int nelem = state.num_elems();
  const int n0 = tl.n0;

  std::vector<int> vector3d_shape = {nelem,2,NP,NP,NUM_PHYSICAL_LEV};
  std::vector<int> scalar3d_shape = {nelem,  NP,NP,NUM_PHYSICAL_LEV};

  check_shape(uv,vector3d_shape);
  check_shape(vthdp,scalar3d_shape);
  check_shape(dp,scalar3d_shape);
  assert ((int)uv.dtype().bits==64);
  assert ((int)vthdp.dtype().bits==64);
  assert ((int)dp.dtype().bits==64);

  ExecViewUnmanaged<double*****> uv_k (vp2dp(uv.data()),nelem,2,NP,NP,NUM_PHYSICAL_LEV);
  ExecViewUnmanaged<double****>  vth_k(vp2dp(vthdp.data()),nelem,NP,NP,NUM_PHYSICAL_LEV);
  ExecViewUnmanaged<double****>  dp_k (vp2dp(dp.data()),nelem,NP,NP,NUM_PHYSICAL_LEV);

  auto v_s = state.m_v;
  auto vtheta_dp_s = state.m_vtheta_dp;
  auto dp3d_s = state.m_dp3d;

  using policy_t = Kokkos::MDRangePolicy<ExecSpace,Kokkos::Rank<4>>;
  policy_t p ({0,0,0,0},{nelem,NP,NP,NUM_PHYSICAL_LEV});
  auto copy = KOKKOS_LAMBDA (int ie, int ip, int jp, int k) {
    int lev = k / VECTOR_SIZE;
    int vec = k % VECTOR_SIZE;
    uv_k(ie,0,ip,jp,k) = ADValue(v_s(ie,n0,0,ip,jp,lev)[vec]);
    uv_k(ie,1,ip,jp,k) = ADValue(v_s(ie,n0,1,ip,jp,lev)[vec]);
    vth_k(ie,ip,jp,k) = ADValue(vtheta_dp_s(ie,n0,ip,jp,lev)[vec]);
    dp_k(ie,ip,jp,k) = ADValue(dp3d_s(ie,n0,ip,jp,lev)[vec]);
  };
  Kokkos::parallel_for(p,copy);
}

void forward(const double dt)
{
  int nm1, n0, np1, nstep;
  prim_run_subcycle_c(dt,nstep,nm1,n0,np1,-1,1);
}

void finalize()
{
  prim_finalize_f90();
}

} // namespace pyhommexx
} // namespace Homme
