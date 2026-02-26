#include "pyhommexx.hpp"

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

} // namespace pyhommexx
