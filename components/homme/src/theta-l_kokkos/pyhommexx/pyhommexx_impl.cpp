#include "pyhommexx_decl.hpp"
#include "pyhommexx_c2f.hpp"

#include "Context.hpp"
#include "CaarFunctor.hpp"
#include "ElementsState.hpp"
#include "ElementsGeometry.hpp"
#include "Hommexx_Session.hpp"
#include "RKStageData.hpp"
#include "SimulationParams.hpp"
#include "TimeLevel.hpp"
#include "Tracers.hpp"
#include "Types.hpp"
#include "mpi/Comm.hpp"
#include "utilities/ViewUtils.hpp"

#include <nanobind/ndarray.h>

#include <fstream>

namespace Homme {
extern "C" {
void prim_run_subcycle_c (const Real& dt, int& nstep, int& nm1, int& n0, int& np1,
                          const int& next_output_step, const int& nsplit_iteration);
}
}

namespace pyhommexx {

using namespace Homme;

struct OutputRedirection {
  void toggle (bool on) {
    if (on) {
      std::cout.rdbuf(cout);
      std::cerr.rdbuf(cerr);
    } else {
      std::cout.rdbuf(blackhole.rdbuf());
      std::cerr.rdbuf(blackhole.rdbuf());
    }
  }
  static OutputRedirection& instance() {
    static OutputRedirection out_red;
    return out_red;
  }
protected:
  OutputRedirection ()
   : cout (std::cout.rdbuf())
   , cerr (std::cerr.rdbuf())
   , blackhole("/dev/null")
  {}

  std::streambuf* cout;
  std::streambuf* cerr;

  std::ofstream blackhole;
};
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

void init_session (const bool do_print_to_screen)
{
  print_to_screen(do_print_to_screen);
  init_parallel_f90();
  // Throw, so we can use try blocks in py
  Session::m_throw_instead_of_abort = true;
}

void print_to_screen (const bool enabled)
{
  print_to_screen_f90 (enabled);
  OutputRedirection::instance().toggle(enabled);
}

void read_params (const nb::str& nml_filename)
{
  auto nml_filename_c = nml_filename.c_str();
  read_params_f90(nml_filename_c);
}
void model_init ()
{
  model_init_f90();
}

nb::dict get_params()
{
  const auto& c = Context::singleton();
  const auto& s = c.get<SimulationParams>();

  nb::dict params;

  params["rsplit"] = s.rsplit;
  params["qsplit"] = s.qsplit;
  params["qsize"] = s.qsize;
  params["qsize_d"] = QSIZE_D;
  params["np"] = NP;
  params["nlev"] = NUM_PHYSICAL_LEV;
  params["hydrostatic"] = s.theta_hydrostatic_mode;
  params["nu"] = s.nu;
  params["ne"] = s.ne;

  return params;
}
void set_params(const nb::dict& params)
{
  using namespace Errors;

  auto& c = Context::singleton();
  auto& s = c.get<SimulationParams>();

  for (const auto& [key,value] : params) {
    runtime_check(nb::isinstance<nb::str>(key),
                  "Error! Functor params dict keys should be strings.\n");

    std::string key_str(nb::cast<nb::str>(key).c_str());
    if (key_str=="alloc_sphere_coords") {
      runtime_check(nb::isinstance<bool>(value),
                    "Error! Functor param 'alloc_sphere_coords' should be a boolean.\n");
      s.alloc_sphere_coords = nb::cast<bool>(value);
    } else {
      runtime_abort("[ERROR] Invalid key for set_params.\n"
                    " - key: " + key_str + "\n" +
                    " - valid keys: alloc_sphere_coords\n");
    }
  }
}

int get_nelemd ()
{
  const auto& c = Context::singleton();
  const auto& e = c.get<ElementsState>();
  return e.num_elems();
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

void get_state_var (nb::ndarray<double>& arr, const nb::str& name)
{
  const auto& c = Context::singleton();
  const auto& state = c.get<ElementsState> ();
  const auto& tracers = c.get<Tracers> ();
  const auto& tl = c.get<TimeLevel> ();

  const int nelem = state.num_elems();
  const int n0 = tl.n0;
  const int qn0 = tl.n0_qdp;

  std::vector<int> vector3dm_shape = {nelem,2,NP,NP,NUM_PHYSICAL_LEV};
  std::vector<int> scalar3dm_shape = {nelem,  NP,NP,NUM_PHYSICAL_LEV};
  std::vector<int> scalar3di_shape = {nelem,  NP,NP,NUM_INTERFACE_LEV};

  constexpr int flag_u   = 0;
  constexpr int flag_v   = 1;
  constexpr int flag_uv  = 2;
  constexpr int flag_vth = 3;
  constexpr int flag_dp  = 4;
  constexpr int flag_w   = 5;
  constexpr int flag_phi = 6;
  constexpr int flag_qv  = 7;
  std::map<std::string,int> which_map = {
    {"u",  flag_u},
    {"v",  flag_v},
    {"uv", flag_uv},
    {"vth",flag_vth},
    {"dp", flag_dp},
    {"w",  flag_w},
    {"phi",flag_phi},
    {"qv", flag_qv},
  };

  // nanobind has no operator== with const char[]
  std::string n (name.c_str());
  if (n=="u" or n=="v" or n=="dp" or n=="qv" or n=="vthetadp") {
    check_shape(arr,scalar3dm_shape);
  } else if (n=="uv") {
    check_shape(arr,vector3dm_shape);
  } else if (n=="phi" or n=="w") {
    check_shape(arr,scalar3di_shape);
  } else {
    throw std::runtime_error("Unrecognized/unsupported state var '" + n + "'.\n");
  }
  assert ((int)arr.dtype().bits==64);

  // They are unmanaged, so we can create all of them, even if we don't use them
  ExecViewUnmanaged<double*****> vec_mid_v (vp2dp(arr.data()),nelem,2,NP,NP,NUM_PHYSICAL_LEV);
  ExecViewUnmanaged<double****>  scl_mid_v (vp2dp(arr.data()),nelem,  NP,NP,NUM_PHYSICAL_LEV);
  ExecViewUnmanaged<double****>  scl_int_v (vp2dp(arr.data()),nelem,  NP,NP,NUM_INTERFACE_LEV);

  auto uv  = state.m_v;
  auto vth = state.m_vtheta_dp;
  auto dp  = state.m_dp3d;
  auto w   = state.m_w_i;
  auto phi = state.m_phinh_i;
  auto qv  = tracers.Q;

  auto which = which_map.at(n);
  int nlev = (which==flag_phi or which==flag_w) ? NUM_INTERFACE_LEV : NUM_PHYSICAL_LEV;
  using policy_t = Kokkos::MDRangePolicy<ExecSpace,Kokkos::Rank<4>>;
  policy_t p ({0,0,0,0},{nelem,NP,NP,nlev});
  auto copy = KOKKOS_LAMBDA (int ie, int ip, int jp, int k) {
    int lev = k / VECTOR_SIZE;
    int vec = k % VECTOR_SIZE;
    switch (which) {
      case flag_u:
        scl_mid_v(ie,ip,jp,k) = ADValue(uv(ie,n0,0,ip,jp,lev)[vec]); break;
      case flag_v:
        scl_mid_v(ie,ip,jp,k) = ADValue(uv(ie,n0,1,ip,jp,lev)[vec]); break;
      case flag_uv:
        vec_mid_v(ie,0,ip,jp,k) = ADValue(uv(ie,n0,0,ip,jp,lev)[vec]);
        vec_mid_v(ie,1,ip,jp,k) = ADValue(uv(ie,n0,1,ip,jp,lev)[vec]); break;
      case flag_vth:
        scl_mid_v(ie,ip,jp,k) = ADValue(vth(ie,n0,ip,jp,lev)[vec]); break;
      case flag_dp:
        scl_mid_v(ie,ip,jp,k) = ADValue(dp(ie,n0,ip,jp,lev)[vec]); break;
      case flag_w:
        scl_int_v(ie,ip,jp,k) = ADValue(w(ie,n0,ip,jp,lev)[vec]); break;
      case flag_phi:
        scl_int_v(ie,ip,jp,k) = ADValue(phi(ie,n0,ip,jp,lev)[vec]); break;
      case flag_qv:
        scl_mid_v(ie,ip,jp,k) = ADValue(qv(ie,qn0,ip,jp,lev)[vec]); break;
      default:
        Kokkos::abort("Unsupported value for 'which' in get_state_var.\n");
    }
  };
  Kokkos::parallel_for(p,copy);
}

void set_state_var (const nb::ndarray<double>& arr, const nb::str& name)
{
  const auto& c = Context::singleton();
  const auto& state = c.get<ElementsState> ();
  const auto& tracers = c.get<Tracers> ();
  const auto& tl = c.get<TimeLevel> ();

  const int nelem = state.num_elems();
  const int n0 = tl.n0;
  const int qn0 = tl.n0_qdp;

  std::vector<int> vector3dm_shape = {nelem,2,NP,NP,NUM_PHYSICAL_LEV};
  std::vector<int> scalar3dm_shape = {nelem,  NP,NP,NUM_PHYSICAL_LEV};
  std::vector<int> scalar3di_shape = {nelem,  NP,NP,NUM_INTERFACE_LEV};

  constexpr int flag_u   = 0;
  constexpr int flag_v   = 1;
  constexpr int flag_uv  = 2;
  constexpr int flag_vth = 3;
  constexpr int flag_dp  = 4;
  constexpr int flag_w   = 5;
  constexpr int flag_phi = 6;
  constexpr int flag_qv  = 7;
  std::map<std::string,int> which_map = {
    {"u",  flag_u},
    {"v",  flag_v},
    {"uv", flag_uv},
    {"vth",flag_vth},
    {"dp", flag_dp},
    {"w",  flag_w},
    {"phi",flag_phi},
    {"qv", flag_qv},
  };

  // nanobind has no operator== with const char[]
  std::string n (name.c_str());
  if (n=="u" or n=="v" or n=="dp" or n=="qv" or n=="vtheta") {
    check_shape(arr,scalar3dm_shape);
  } else if (n=="uv") {
    check_shape(arr,vector3dm_shape);
  } else if (n=="phi" or n=="w") {
    check_shape(arr,scalar3di_shape);
  } else {
    throw std::runtime_error("Unrecognized/unsupported state var '" + n + "'.\n");
  }
  assert ((int)arr.dtype().bits==64);

  // They are unmanaged, so we can create all of them, even if we don't use them
  ExecViewUnmanaged<const double*****> vec_mid_v (vp2cdp(arr.data()),nelem,2,NP,NP,NUM_PHYSICAL_LEV);
  ExecViewUnmanaged<const double****>  scl_mid_v (vp2cdp(arr.data()),nelem,  NP,NP,NUM_PHYSICAL_LEV);
  ExecViewUnmanaged<const double****>  scl_int_v (vp2cdp(arr.data()),nelem,  NP,NP,NUM_INTERFACE_LEV);

  auto uv  = state.m_v;
  auto vth = state.m_vtheta_dp;
  auto dp  = state.m_dp3d;
  auto w   = state.m_w_i;
  auto phi = state.m_phinh_i;
  auto qv  = tracers.Q;

  auto which = which_map.at(n);
  int nlev = (which==flag_phi or which==flag_w) ? NUM_INTERFACE_LEV : NUM_PHYSICAL_LEV;
  using policy_t = Kokkos::MDRangePolicy<ExecSpace,Kokkos::Rank<4>>;
  policy_t p ({0,0,0,0},{nelem,NP,NP,nlev});
  auto copy = KOKKOS_LAMBDA (int ie, int ip, int jp, int k) {
    int lev = k / VECTOR_SIZE;
    int vec = k % VECTOR_SIZE;
    switch (which) {
      case flag_u:
        uv(ie,n0,0,ip,jp,lev)[vec] = scl_mid_v(ie,ip,jp,k); break;
      case flag_v:
        uv(ie,n0,1,ip,jp,lev)[vec] = scl_mid_v(ie,ip,jp,k); break;
      case flag_uv:
        uv(ie,n0,0,ip,jp,lev)[vec] = vec_mid_v(ie,0,ip,jp,k);
        uv(ie,n0,1,ip,jp,lev)[vec] = vec_mid_v(ie,1,ip,jp,k); break;
      case flag_vth:
        vth(ie,n0,ip,jp,lev)[vec] = scl_mid_v(ie,ip,jp,k); break;
      case flag_dp:
        dp(ie,n0,ip,jp,lev)[vec] = scl_mid_v(ie,ip,jp,k); break;
      case flag_w:
        w(ie,n0,ip,jp,lev)[vec] = scl_int_v(ie,ip,jp,k); break;
      case flag_phi:
        phi(ie,n0,ip,jp,lev)[vec] = scl_int_v(ie,ip,jp,k); break;
      case flag_qv:
        qv(ie,n0,ip,jp,lev)[vec] = scl_mid_v(ie,ip,jp,k); break;
      default:
        Kokkos::abort("Unsupported value for 'which' in get_state_var.\n");
    }
  };
  Kokkos::parallel_for(p,copy);
}

void get_state_var_sens (nb::ndarray<double>& arr, const nb::str& name)
{
#ifdef HOMMEXX_ENABLE_FAD_TYPES
  const auto& c = Context::singleton();
  const auto& state = c.get<ElementsState> ();
  const auto& tracers = c.get<Tracers> ();
  const auto& tl = c.get<TimeLevel> ();

  const int nelem = state.num_elems();
  const int n0 = tl.n0;
  const int qn0 = tl.n0_qdp;

  std::vector<int> vector3dm_shape = {nelem,2,NP,NP,NUM_PHYSICAL_LEV,HOMMEXX_SFAD_SIZE};
  std::vector<int> scalar3dm_shape = {nelem,  NP,NP,NUM_PHYSICAL_LEV,HOMMEXX_SFAD_SIZE};
  std::vector<int> scalar3di_shape = {nelem,  NP,NP,NUM_INTERFACE_LEV,HOMMEXX_SFAD_SIZE};

  constexpr int flag_u   = 0;
  constexpr int flag_v   = 1;
  constexpr int flag_uv  = 2;
  constexpr int flag_vth = 3;
  constexpr int flag_dp  = 4;
  constexpr int flag_w   = 5;
  constexpr int flag_phi = 6;
  constexpr int flag_qv  = 7;
  std::map<std::string,int> which_map = {
    {"u",  flag_u},
    {"v",  flag_v},
    {"uv", flag_uv},
    {"vth",flag_vth},
    {"dp", flag_dp},
    {"w",  flag_w},
    {"phi",flag_phi},
    {"qv", flag_qv},
  };

  // nanobind has no operator== with const char[]
  std::string n (name.c_str());
  if (n=="u" or n=="v" or n=="dp" or n=="qv" or n=="vthetadp") {
    check_shape(arr,scalar3dm_shape);
  } else if (n=="uv") {
    check_shape(arr,vector3dm_shape);
  } else if (n=="phi" or n=="w") {
    check_shape(arr,scalar3di_shape);
  } else {
    throw std::runtime_error("Unrecognized/unsupported state var '" + n + "'.\n");
  }
  assert ((int)arr.dtype().bits==64);

  // They are unmanaged, so we can create all of them, even if we don't use them
  ExecViewUnmanaged<double******> vec_mid_v (vp2dp(arr.data()),nelem,2,NP,NP,NUM_PHYSICAL_LEV);
  ExecViewUnmanaged<double*****>  scl_mid_v (vp2dp(arr.data()),nelem,  NP,NP,NUM_PHYSICAL_LEV);
  ExecViewUnmanaged<double*****>  scl_int_v (vp2dp(arr.data()),nelem,  NP,NP,NUM_INTERFACE_LEV);

  auto uv  = state.m_v;
  auto vth = state.m_vtheta_dp;
  auto dp  = state.m_dp3d;
  auto w   = state.m_w_i;
  auto phi = state.m_phinh_i;
  auto qv  = tracers.Q;

  auto which = which_map.at(n);
  int nlev = (which==flag_phi or which==flag_w) ? NUM_INTERFACE_LEV : NUM_PHYSICAL_LEV;
  using policy_t = Kokkos::MDRangePolicy<ExecSpace,Kokkos::Rank<5>>;
  policy_t p ({0,0,0,0,0},{nelem,NP,NP,nlev,HOMMEXX_SFAD_SIZE});
  auto copy = KOKKOS_LAMBDA (int ie, int ip, int jp, int k, int ider) {
    int lev = k / VECTOR_SIZE;
    int vec = k % VECTOR_SIZE;
    switch (which) {
      case flag_u:
        scl_mid_v(ie,ip,jp,k,ider) = uv(ie,n0,0,ip,jp,lev)[vec].fastAccessDx(ider); break;
      case flag_v:
        scl_mid_v(ie,ip,jp,k,ider) = uv(ie,n0,1,ip,jp,lev)[vec].fastAccessDx(ider); break;
      case flag_uv:
        vec_mid_v(ie,0,ip,jp,k,ider) = uv(ie,n0,0,ip,jp,lev)[vec].fastAccessDx(ider);
        vec_mid_v(ie,1,ip,jp,k,ider) = uv(ie,n0,1,ip,jp,lev)[vec].fastAccessDx(ider); break;
      case flag_vth:
        scl_mid_v(ie,ip,jp,k,ider) = vth(ie,n0,ip,jp,lev)[vec].fastAccessDx(ider); break;
      case flag_dp:
        scl_mid_v(ie,ip,jp,k,ider) = dp(ie,n0,ip,jp,lev)[vec].fastAccessDx(ider); break;
      case flag_w:
        scl_int_v(ie,ip,jp,k,ider) = w(ie,n0,ip,jp,lev)[vec].fastAccessDx(ider); break;
      case flag_phi:
        scl_int_v(ie,ip,jp,k,ider) = phi(ie,n0,ip,jp,lev)[vec].fastAccessDx(ider); break;
      case flag_qv:
        scl_mid_v(ie,ip,jp,k,ider) = qv(ie,qn0,ip,jp,lev)[vec].fastAccessDx(ider); break;
      default:
        Kokkos::abort("Unsupported value for 'which' in get_state_var.\n");
    }
  };
  Kokkos::parallel_for(p,copy);
#else
  Errors::runtime_error("pyhommexx::get_state_var_sens not implemented unless HOMMEXX_ENABLE_FAD_TYPES is defined.\n");
  (void)arr;
  (void)name;
#endif
}

void forward(const double dt)
{
  int nm1, n0, np1, nstep;
  prim_run_subcycle_c(dt,nstep,nm1,n0,np1,-1,1);
}

void run_functor(const nb::str& name, const nb::dict& params)
{
  using namespace Errors;
  std::string name_str (name.c_str());
  if (name_str=="caar") {
    auto& c = Context::singleton();
    auto& f = c.get<CaarFunctor>();
    auto& tl = c.get<TimeLevel>();

    RKStageData data;

    data.nm1 = tl.nm1;
    data.n0  = tl.n0;
    data.np1 = tl.np1;
    data.n0_qdp = 0;
    data.scale1 = 1;
    data.scale2 = 0;
    data.scale3 = 1;
    data.eta_ave_w = 0;

    bool update_tl = false;
    for (const auto& [key, value] : params) {
      runtime_check(nb::isinstance<nb::str>(key),
                    "Error! Functor params dict keys should be strings.\n");

      std::string key_str(nb::cast<nb::str>(key).c_str());
      if (key_str=="dt") {
        runtime_check(nb::isinstance<double>(value),
                      "Error! Functor param 'dt' should be a double.\n");
        data.dt = nb::cast<double>(value);
      } else if (key_str=="eta_ave_w") {
        runtime_check(nb::isinstance<double>(value),
                      "Error! Functor param 'eta_ave_w' should be a double.\n");
        data.eta_ave_w = nb::cast<double>(value);
      } else if (key_str=="scale1") {
        runtime_check(nb::isinstance<double>(value),
                      "Error! Functor param 'scale1' should be a double.\n");
        data.scale1 = nb::cast<double>(value);
      } else if (key_str=="scale2") {
        runtime_check(nb::isinstance<double>(value),
                      "Error! Functor param 'scale2' should be a double.\n");
        data.scale2 = nb::cast<double>(value);
      } else if (key_str=="scale3") {
        runtime_check(nb::isinstance<double>(value),
                      "Error! Functor param 'scale3' should be a double.\n");
        data.scale3 = nb::cast<double>(value);
      } else if (key_str=="update_tl") {
        runtime_check(nb::isinstance<bool>(value),
                      "Error! Functor param 'update_tl' should be a bool.\n");
        update_tl = nb::cast<bool>(value);
      } else {
        runtime_abort("[ERROR] Invalid key for caar functor params.\n"
                      " - key: " + key_str + "\n" +
                      " - valid keys: dt\n");
      }
    }

    std::cout << "running functor " << name_str << "\n";

    f.run(data);

    if (update_tl) {
      tl.update_dynamics_levels(UpdateType::FORWARD);
    }
  } else {
    runtime_abort("[ERROR] Unrecognized/unsupported fucntor name.\n"
                  " - input name: " + name_str + "\n"
                  " - valid name(s): caar\n");
  }
}

void finalize()
{
  prim_finalize_f90();
}

} // namespace pyhommexx
