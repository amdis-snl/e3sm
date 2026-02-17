/********************************************************************************
 * HOMMEXX 1.0: Copyright of Sandia Corporation
 * This software is released under the BSD license
 * See the file 'COPYRIGHT' in the HOMMEXX/src/share/cxx directory
 *******************************************************************************/

#ifndef HOMMEXX_VIEW_UTILS_HPP
#define HOMMEXX_VIEW_UTILS_HPP

#include "Types.hpp"
#include "ExecSpaceDefs.hpp"

namespace Homme {

// This helper struct (and its shorter alias) simply provide
// the map 'Scalar->Real' and 'const Scalar->const Real'
namespace Impl {
template<typename In, typename Out>
struct ConstIfConst {
  static constexpr bool is_const = std::is_const<In>::value;
  using type = typename std::conditional<is_const,
                  typename std::add_const<Out>::type,
                  typename std::remove_const<Out>::type
               >::type;
};
} // namespace Impl

// Structure to define the type of a view that has a const data type,
// given the type of an input view
template<typename ViewT>
struct ViewConst{};

// Note:: using ViewType::const_type may add explicit template arguments to ViewType.
//        Instead, we want the same 'template signature', simply with a const data type.
template<typename DataType, typename...Props>
struct ViewConst<ViewType<DataType,Props...>> {
  using const_dtype = typename ViewType<DataType,Props...>::traits::const_data_type;
  using type = ViewType<const_dtype,Props...>;
};

// This is ugly, but prevents unnecessary copies
template<typename ViewT>
KOKKOS_INLINE_FUNCTION
typename ViewConst<ViewT>::type
viewConst(const ViewT& v) {
  return reinterpret_cast<const typename ViewConst<ViewT>::type&>(v);
}

template<int NUM_LEVELS>
KOKKOS_INLINE_FUNCTION
void print_col (const char* prefix,
                const ExecViewUnmanaged<const Scalar[ColInfo<NUM_LEVELS>::NumPacks]>& v) {
  printf("%s:",prefix);
  for (int k=0; k<NUM_LEVELS; ++k) {
    const int ilev = k / VECTOR_SIZE;
    const int ivec = k % VECTOR_SIZE;
    printf(" %3.15f",v(ilev)[ivec]);
  }
  printf("\n");
}

template<int NUM_LEVELS>
KOKKOS_INLINE_FUNCTION
void print_col (const char* prefix, const ExecViewUnmanaged<const Real[NUM_LEVELS]>& v) {
  printf("%s:",prefix);
  for (int k=0; k<NUM_LEVELS; ++k) {
    printf(" %3.15f",v(k));
  }
  printf("\n");
}

template<typename ViewT>
int has_nans (const ViewT& v)
{
  int dim0 = v.extent(0);
  int dim1 = v.extent(1);
  int dim2 = v.extent(2);
  int dim3 = v.extent(3);
  int dim4 = v.extent(4);
  int dim5 = v.extent(5);
  auto vh = Kokkos::create_mirror_view(v);
  Kokkos::deep_copy(vh,v);
  constexpr int N = v.rank;
  if constexpr (N==0) {
    return Homme::isnan(ADValue(vh()));
  } else if constexpr (N==1) {
    for (int i=0;i<dim0;++i)
      if (Homme::isnan(ADValue(vh(i))))
        return i;
  } else if constexpr (N==2) {
    for (int i=0; i<dim0; ++i)
      for (int j=0; j<dim1; ++j)
        if (Homme::isnan(ADValue(vh(i,j))))
          return i*dim1+j;
  } else if constexpr (N==3) {
    for (int i=0; i<dim0; ++i)
      for (int j=0; j<dim1; ++j)
        for (int k=0; k<dim2; ++k)
          if (Homme::isnan(ADValue(vh(i,j,k))))
            return i*dim1*dim2+j*dim2+k;
  } else if constexpr (N==4) {
    for (int i=0; i<dim0; ++i)
      for (int j=0; j<dim1; ++j)
        for (int k=0; k<dim2; ++k)
          for (int l=0; l<dim3; ++l)
            if (Homme::isnan(ADValue(vh(i,j,k,l))))
              return i*dim1*dim2*dim3+j*dim2*dim3+k*dim3+l;
  } else if constexpr (N==5) {
    for (int i=0; i<dim0; ++i)
      for (int j=0; j<dim1; ++j)
        for (int k=0; k<dim2; ++k)
          for (int l=0; l<dim3; ++l)
            for (int m=0; m<dim4; ++m)
              if (Homme::isnan(ADValue(vh(i,j,k,l,m))))
                return i*dim1*dim2*dim3*dim4+j*dim2*dim3*dim4+k*dim3*dim4+l*dim4+m;
  } else if constexpr (N==6) {
    for (int i=0; i<dim0; ++i)
      for (int j=0; j<dim1; ++j)
        for (int k=0; k<dim2; ++k)
          for (int l=0; l<dim3; ++l)
            for (int m=0; m<dim4; ++m)
              for (int n=0; n<dim5; ++n)
                if (Homme::isnan(ADValue(vh(i,j,k,l,m,n))))
                  return i*dim1*dim2*dim3*dim4*dim5+j*dim2*dim3*dim4*dim5+k*dim3*dim4*dim5+l*dim4*dim5+m*dim5+n;
  } else {
    throw std::runtime_error("[has_nans] Unsupported rank");
  }
  return -1;
}

} // namespace Homme

#endif // HOMMEXX_VIEW_UTILS_HPP
