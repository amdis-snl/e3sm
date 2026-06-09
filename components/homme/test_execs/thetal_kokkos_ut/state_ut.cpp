#include <catch2/catch.hpp>

#include "ElementsState.hpp"
#include "ElementsState_def.hpp"

#include "Types.hpp"

#include "utilities/TestUtils.hpp"

#include <ekat_pack_kokkos.hpp>

#include <iostream>
#include <random>

using namespace Homme;

TEST_CASE("state_tests")
{
  using rngAlg = std::mt19937_64;
  using rpdf = std::uniform_real_distribution<Real>;
  using ipdf = std::uniform_int_distribution<int>;

  std::random_device rd;
  const unsigned int catchRngSeed = Catch::rngSeed();
  static unsigned int seed;
  static bool printed = false;
  if (not printed) {
    seed = catchRngSeed==0 ? rd() : catchRngSeed;
    std::cout << "seed: " << seed << (catchRngSeed==0 ? " (catch rng seed was 0)\n" : "\n");
  }
  rngAlg engine(seed);

  constexpr int num_elems = 4;
  constexpr auto A = Kokkos::ALL();

  SECTION ("export") {
    ElementsStateST<Real> state;
    state.init(num_elems);
    state.randomize(seed);

    auto slice0 = state.take_snapshot(0,false);
    auto slice1 = state.take_snapshot(1,true);

    REQUIRE (slice0.ps_v.data()==nullptr);
    REQUIRE (slice1.ps_v.data()!=nullptr);

    REQUIRE (slice0.v.data()!=state.m_v.data());
    REQUIRE (slice0.v.extent(0)==num_elems);

    REQUIRE (views_are_equal(ekat::scalarize(slice0.v),Kokkos::subview(ekat::scalarize(state.m_v),A,0,A,A,A,A),NUM_PHYSICAL_LEV));
    REQUIRE (views_are_equal(ekat::scalarize(slice1.ps_v),Kokkos::subview(ekat::scalarize(state.m_ps_v),A,0,A,A)));

    // Make sure we are not getting false positives
    REQUIRE_FALSE(views_are_equal(ekat::scalarize(slice0.vtheta_dp),ekat::scalarize(slice0.dp3d),NUM_PHYSICAL_LEV));
  }

#ifdef HOMMEXX_ENABLE_FAD_TYPES
  SECTION ("import_from_deriv") {
    using FadT = SFadN<Real,2>;
    ElementsStateST<Real> state_real;
    state_real.init(num_elems);

    ElementsStateST<FadT> state_fad;
    state_fad.init(num_elems);
    
    int tl =2;
    int ider = 1;
    state_real.import_values_from_deriv (state_fad,tl,ider);
    auto vrh = Kokkos::create_mirror_view(ekat::scalarize(state_real.m_v));
    auto vfh = Kokkos::create_mirror_view(ekat::scalarize(state_fad.m_v));
    Kokkos::deep_copy(vrh,ekat::scalarize(state_real.m_v));
    Kokkos::deep_copy(vfh,ekat::scalarize(state_fad.m_v));
    for (int ie=0; ie<num_elems; ++ie)
      for (int ip=0; ip<NP; ++ip)
        for (int jp=0; jp<NP; ++jp)
          for (int k=0; k<NUM_PHYSICAL_LEV; ++k) {
            REQUIRE (vrh(ie,tl,0,ip,jp,k)==vfh(ie,tl,0,ip,jp,k).fastAccessDx(ider));
            REQUIRE (vrh(ie,tl,1,ip,jp,k)==vfh(ie,tl,1,ip,jp,k).fastAccessDx(ider));
          }
  }
#endif
}
