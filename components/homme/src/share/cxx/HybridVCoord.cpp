/********************************************************************************
 * HOMMEXX 1.0: Copyright of Sandia Corporation
 * This software is released under the BSD license
 * See the file 'COPYRIGHT' in the HOMMEXX/src/share/cxx directory
 *******************************************************************************/

#include "HybridVCoord.hpp"
#include "ColumnOps.hpp"
#include "ErrorDefs.hpp"
#include "PhysicalConstants.hpp"
#include "utilities/TestUtils.hpp"
#include "utilities/ViewUtils.hpp"

#include <random>

namespace Homme
{

void HybridVCoord::init(const Real ps0_in,
                   CRCPtr hybrid_am_ptr,
                   CRCPtr hybrid_ai_ptr,
                   CRCPtr hybrid_bm_ptr,
                   CRCPtr hybrid_bi_ptr)
{
  // Sanity checks
  assert(hybrid_am_ptr!=nullptr);
  assert(hybrid_ai_ptr!=nullptr);
  assert(hybrid_bm_ptr!=nullptr);
  assert(hybrid_bi_ptr!=nullptr);

  ps0 = ps0_in;

  // Create and init interface views
  hybrid_ai = ExecViewManaged<Real[NUM_INTERFACE_LEV]>("Hybrid coordinates; coefficient A_interfaces");
  hybrid_bi = ExecViewManaged<Real[NUM_INTERFACE_LEV]>("Hybrid coordinates; coefficient B_interfaces");
  HostViewUnmanaged<const Real[NUM_INTERFACE_LEV]> host_hybrid_ai(hybrid_ai_ptr);
  HostViewUnmanaged<const Real[NUM_INTERFACE_LEV]> host_hybrid_bi(hybrid_bi_ptr);
  Kokkos::deep_copy(hybrid_ai, host_hybrid_ai);
  Kokkos::deep_copy(hybrid_bi, host_hybrid_bi);

  hybrid_ai_packed = ExecViewManaged<Scalar[NUM_LEV_P]>("Hybrid coordinates; coefficient A_interfaces");
  hybrid_bi_packed = ExecViewManaged<Scalar[NUM_LEV_P]>("Hybrid coordinates; coefficient B_interfaces");
  auto hyai_p_h = Kokkos::create_mirror_view(hybrid_ai_packed);
  auto hybi_p_h = Kokkos::create_mirror_view(hybrid_bi_packed);
  for (int i=0; i<NUM_INTERFACE_LEV; ++i) {
    int ilev = i / VECTOR_SIZE;
    int ivec = i % VECTOR_SIZE;
    hyai_p_h(ilev)[ivec] = host_hybrid_ai(i);
    hybi_p_h(ilev)[ivec] = host_hybrid_bi(i);
  }

  // Create and init midpoints views
  hybrid_am = ExecViewManaged<Scalar[NUM_LEV]>("Hybrid coordinates; coefficient A_midpoints");
  hybrid_bm = ExecViewManaged<Scalar[NUM_LEV]>("Hybrid coordinates; coefficient B_midpoints");

  HostViewUnmanaged<const Real[NUM_PHYSICAL_LEV]> host_hybrid_am(hybrid_am_ptr);
  HostViewUnmanaged<const Real[NUM_PHYSICAL_LEV]> host_hybrid_bm(hybrid_bm_ptr);

  auto hyam_unpacked = unpackView(hybrid_am);
  auto hybm_unpacked = unpackView(hybrid_bm);
  auto hyam_unpacked_h = Kokkos::create_mirror_view(hyam_unpacked);
  auto hybm_unpacked_h = Kokkos::create_mirror_view(hybm_unpacked);
  for (int i=0; i<NUM_PHYSICAL_LEV; ++i) {
    hyam_unpacked_h(i) = host_hybrid_am(i);
    hybm_unpacked_h(i) = host_hybrid_bm(i);
  }
  Kokkos::deep_copy(hyam_unpacked,hyam_unpacked_h);
  Kokkos::deep_copy(hybm_unpacked,hybm_unpacked_h);

  // i don't think this saves us much now
  {
    // Only hybrid_ai(0) is needed.
    hybrid_ai0 = hybrid_ai_ptr[0];
  }

  // Compute delta_A and delta_B
  compute_deltas();
  // Compute Eta=A+B
  compute_eta();

  m_inited = true;
}

void HybridVCoord::random_init(int seed) {
  hybrid_ai_packed = ExecViewManaged<Scalar[NUM_LEV_P]>("Hybrid coordinates; coefficient A_interfaces");
  hybrid_bi_packed = ExecViewManaged<Scalar[NUM_LEV_P]>("Hybrid coordinates; coefficient B_interfaces");

  hybrid_ai = ExecViewManaged<Real[NUM_INTERFACE_LEV]>("Hybrid coordinates; coefficient A_interfaces");
  hybrid_bi = ExecViewManaged<Real[NUM_INTERFACE_LEV]>("Hybrid coordinates; coefficient B_interfaces");

  hybrid_am = ExecViewManaged<Scalar[NUM_LEV]>("Hybrid a_interface coefs");
  hybrid_bm = ExecViewManaged<Scalar[NUM_LEV]>("Hybrid b_interface coefs");

  std::mt19937_64 engine(seed);
  ps0 = 1.0;

  using urd = std::uniform_real_distribution<Real>;

  // eta=hybrid_a+hybrid_b increasing ensures p is monotonic (sufficient, not necessary cond).
  // Also, hybrid_a and hybrid_b must be between 0 and 1.
  // So easy way: generate uniform eta=a+b in [eta_min,1], with eta_min>0.
  // That means eta(i) = i*delta_eta, with delta_eta = (1-eta_min)/NUM_PHYSICAL_LEV.
  // Then pick a = c*b, with c<1 (say c=0.1). Then eta = b+c*b=(c+1)b,
  // so b(i) = i*delta_eta/(c+1) and a(i) = c*i*delta_eta/(c+1).
  // To add randomness, perturb sligthly: a(i)=a(i)+da and b(i)=b(i)+db,
  // without violating eta(i+1)>eta(i). For instance, let da+db<delta_eta/2,
  // which is true if da<delta_eta/4 and db<delta_eta/4.

  decltype(hybrid_ai)::HostMirror host_hybrid_ai("Host hybrid ai coefs");
  decltype(hybrid_bi)::HostMirror host_hybrid_bi("Host hybrid bi coefs");
  decltype(hybrid_am)::HostMirror host_hybrid_am("Host hybrid am coefs");
  decltype(hybrid_bm)::HostMirror host_hybrid_bm("Host hybrid bm coefs");

  // This somewhat odd expression ensures that deta (below) is smaller
  // than eta_min, so that eta_min + eps*deta is always positive
  constexpr Real eta_min = (1.0/NUM_PHYSICAL_LEV)+0.001;

  // The proportionality constant between a and b.
  Real c;
  genRandArray(&c,1,engine,urd(0.0001,0.25));

  // The eta uniform grid spacing
  Real deta = (1-eta_min)/NUM_PHYSICAL_LEV;
  Real perturbation = deta/5;

  for (int i=0; i<NUM_INTERFACE_LEV; ++i) {
    // Compute eta
    Real eta_i = eta_min+deta*i;

    // Create b and a
    host_hybrid_bi(i) = eta_i/(c+1);
    host_hybrid_ai(i) = c*host_hybrid_bi(i);

    // Add some randomness to a and b
    Real dab[2];
    Real perturbation_adjusted = std::min(std::min(perturbation,eta_i),std::min(perturbation,1-eta_i));;

    auto perturbation_dist = urd(-perturbation_adjusted,perturbation_adjusted);
    genRandArray(dab,2,engine,perturbation_dist);
    host_hybrid_ai(i) += dab[0];
    host_hybrid_bi(i) += dab[1];

    // Enforce 0<a,b<1
    if (host_hybrid_ai(i)<0) {
      host_hybrid_ai(i) = 0;
    }
    if (host_hybrid_bi(i)<0) {
      host_hybrid_bi(i) = 0;
    }
    if (host_hybrid_ai(i)>1) {
      host_hybrid_ai(i) = 1;
    }
    if (host_hybrid_bi(i)>1) {
      host_hybrid_bi(i) = 1;
    }
  }

  // Enforce b(0)=0, a(0)=eta_min, and b(end)=1, a(end)=0;
  host_hybrid_ai(0) = eta_min;
  host_hybrid_bi(0) = 0;
  host_hybrid_ai(NUM_PHYSICAL_LEV) = 0;
  host_hybrid_bi(NUM_PHYSICAL_LEV) = 1;
  
  // Safety check: a+b should be monotone here
  // Also, set midpoints
  HostViewUnmanaged<Real[NUM_PHYSICAL_LEV]> host_hybrid_am_real(reinterpret_cast<Real*>(host_hybrid_am.data()));
  HostViewUnmanaged<Real[NUM_PHYSICAL_LEV]> host_hybrid_bm_real(reinterpret_cast<Real*>(host_hybrid_bm.data()));
  for (int i=1; i<NUM_INTERFACE_LEV; ++i) {
    Real curr = ADValue(host_hybrid_ai(i) + host_hybrid_bi(i));
    Real prev = ADValue(host_hybrid_ai(i-1) + host_hybrid_bi(i-1));   

    Errors::runtime_check(curr>prev,"Error! hybrid_a+hybrid_b is not increasing.\n", -1);

    host_hybrid_am_real(i-1) = ADValue(host_hybrid_ai(i) + host_hybrid_ai(i-1))/2.0;
    host_hybrid_bm_real(i-1) = ADValue(host_hybrid_bi(i) + host_hybrid_bi(i-1))/2.0;
  }

  Kokkos::deep_copy(hybrid_ai, host_hybrid_ai);
  Kokkos::deep_copy(hybrid_bi, host_hybrid_bi);
  Kokkos::deep_copy(hybrid_am, host_hybrid_am);
  Kokkos::deep_copy(hybrid_bm, host_hybrid_bm);

  auto hyai_packed_h = Kokkos::create_mirror_view(hybrid_ai_packed);
  auto hybi_packed_h = Kokkos::create_mirror_view(hybrid_bi_packed);
  for (int i=0; i<NUM_INTERFACE_LEV; ++i) {
    hyai_packed_h(i) = host_hybrid_ai(i);
    hybi_packed_h(i) = host_hybrid_bi(i);
  }
  Kokkos::deep_copy(hybrid_ai_packed,hyai_packed_h);
  Kokkos::deep_copy(hybrid_bi_packed,hybi_packed_h);

  hybrid_ai0 = ADValue(host_hybrid_ai(0));

  compute_deltas();
  compute_eta();

  m_inited = true;
}

void HybridVCoord::compute_deltas ()
{
  // This is obviously not for speed (tiny amount of work, and only at setup),
  // but rather to delegate the logic to a single place (namely ColumnOps),
  // so that we avoid making mistakes by replicating the same algorithm
  hybrid_ai_delta = ExecViewManaged<Scalar[NUM_LEV]>("delta hyai");
  hybrid_bi_delta = ExecViewManaged<Scalar[NUM_LEV]>("delta hybi");
  dp0 = ExecViewManaged<Scalar[NUM_LEV]>("dp0");

  // Create local copies, to avoid issue of lambda on GPU
  auto hyai = hybrid_ai_packed;
  auto hybi = hybrid_bi_packed;
  auto dhyai = hybrid_ai_delta;
  auto dhybi = hybrid_bi_delta;
  auto ldp0 = dp0;
  auto lps0 = ps0;
  auto policy = Homme::get_default_team_policy<ExecSpace>(1);
  Kokkos::parallel_for("[HybridVCoord::compute_deltas]",policy,
                       KOKKOS_LAMBDA(const TeamMember& team) {
    KernelVariables kv(team);
    Kokkos::parallel_for(Kokkos::TeamThreadRange(team,1),[&](int /*unused*/){
      ColumnOps::compute_midpoint_delta(kv,hyai,dhyai);
      ColumnOps::compute_midpoint_delta(kv,hybi,dhybi);
      Kokkos::parallel_for(Kokkos::ThreadVectorRange(kv.team,NUM_LEV),
                           [&](const int ilev) {
        ldp0(ilev) = dhyai(ilev) * lps0 + dhybi(ilev) * lps0;
      });
    });
  });
}

void HybridVCoord::compute_eta ()
{
  // Create device views
  etai = ExecViewManaged<Real[NUM_INTERFACE_LEV]>("Eta coordinate at interfaces");
  etam = ExecViewManaged<Scalar[NUM_LEV]>("Eta coordinate at midpoints");
  exner0 = ExecViewManaged<Scalar[NUM_LEV]>("exner0");

  // Local copies, to avoid issues on GPU when accessing this-> members
  auto l_etai = etai;
  auto l_etam = etam;
  auto l_exner0 = exner0;
  auto l_hybrid_am = hybrid_am;
  auto l_hybrid_bm = hybrid_bm;
  auto l_hybrid_ai = hybrid_ai;
  auto l_hybrid_bi = hybrid_bi;
  const auto l_ps0 = ps0;
  const auto p0 = PhysicalConstants::p0;
  const auto kappa = PhysicalConstants::kappa;

  Kokkos::parallel_for(Kokkos::RangePolicy<ExecSpace>(0,NUM_LEV),
                       KOKKOS_LAMBDA(const int& ilev){
    l_etam(ilev) = l_hybrid_am(ilev) + l_hybrid_bm(ilev);
    l_exner0(ilev) = pow(l_etam(ilev)*l_ps0/p0,kappa);
  });
  Kokkos::parallel_for(Kokkos::RangePolicy<ExecSpace>(0,NUM_INTERFACE_LEV),
                       KOKKOS_LAMBDA(const int& ilev){
    l_etai(ilev) = ADValue(l_hybrid_ai(ilev) + l_hybrid_bi(ilev));
  });
}

} // namespace Homme
