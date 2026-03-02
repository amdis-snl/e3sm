#include "pyhommexx.hpp"

#include <vector>

namespace pyhommexx {

inline double* vp2dp (void* p)
{
  return reinterpret_cast<double*>(p);
}
inline const double* vp2cdp (void* p)
{
  return reinterpret_cast<const double*>(p);
}

template<typename T>
void check_shape(const nb::ndarray<T>& arr, const std::vector<int>& shape)
{
  assert (arr.ndim()==shape.size());
  for (size_t i=0; i<shape.size(); ++i) {
    assert (static_cast<int>(arr.shape(i))==shape[i]);
  }
}

} // namespace pyhommexx
