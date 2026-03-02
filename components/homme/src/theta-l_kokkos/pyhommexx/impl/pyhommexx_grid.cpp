#include "pyhommexx.hpp"
#include "pyhommexx_utils.hpp"

#include "Context.hpp"
#include "ElementsGeometry.hpp"

#include <nanobind/ndarray.h>

extern "C" {
void get_num_unique_pts_f90 (int*& num_per_elem);
void get_unique_pts_f90 (int*& ia_ptr,int*& ja_ptr);
}

namespace pyhommexx {

using namespace Homme;

int get_nelemd ()
{
  const auto& c = Context::singleton();
  const auto& g = c.get<ElementsGeometry>();
  return g.num_elems();
}

void get_num_unique_pts (nb::ndarray<int>& n)
{
  const auto& c = Context::singleton();
  const auto& geo = c.get<ElementsGeometry> ();
  int num_elems = geo.num_elems();
  assert(n.ndim()==1);
  assert(n.shape(0)==num_elems);
  int* data = n.data();
  get_num_unique_pts_f90(data);
}

void get_unique_pts (nb::ndarray<int>& ia,
                     nb::ndarray<int>& ja)
{
  const auto& c = Context::singleton();
  const auto& geo = c.get<ElementsGeometry> ();
  int num_elems = geo.num_elems();
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

void get_dyn_latlon (nb::ndarray<double>& lat,
                     nb::ndarray<double>& lon)
{
  const auto& c = Context::singleton();
  const auto& geo = c.get<ElementsGeometry> ();
  const int nelem = geo.num_elems();

  auto latlon_h = Kokkos::create_mirror_view(geo.m_sphere_latlon);
  Kokkos::deep_copy(latlon_h,geo.m_sphere_latlon);

  ExecViewUnmanaged<double***> lat_v (vp2dp(lat.data()),nelem,NP,NP);
  ExecViewUnmanaged<double***> lon_v (vp2dp(lon.data()),nelem,NP,NP);

  for (int ie=0; ie<nelem; ++ie) {
    for (int ip=0; ip<NP; ++ip) {
      for (int jp=0; jp<NP; ++jp) {
        lat_v(ie,ip,jp) = latlon_h(ie,ip,jp,0);
        lon_v(ie,ip,jp) = latlon_h(ie,ip,jp,1);
      }
    }
  }
}

} // namespace pyhommexx
