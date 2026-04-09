/********************************************************************************
 * HOMMEXX 1.0: Copyright of Sandia Corporation
 * This software is released under the BSD license
 * See the file 'COPYRIGHT' in the HOMMEXX/src/share/cxx directory
 *******************************************************************************/

#include "BoundaryExchangeBase.hpp"

#include "Connectivity.hpp"
#include "ErrorDefs.hpp"
#include "Hommexx_Debug.hpp"
#include "MpiBuffersManager.hpp"

namespace Homme
{

// =============================== BASE CLASS IMPLEMENTATION ============================= //

BoundaryExchangeBase::
~BoundaryExchangeBase ()
{
  // It may be that we never really used this object, and never set the BM...
  if (m_buffers_manager) {
    // Remove me as a customer of the BM
    m_buffers_manager->remove_customer(this);
  }
}

void BoundaryExchangeBase::
set_connectivity (const std::shared_ptr<Connectivity>& connectivity)
{
  // Functionality only available before registration starts
  assert (!m_registration_started && !m_registration_completed);

  // Make sure it is a valid connectivity (does not need to be initialized/finalized yet)
  // Also, replacing the connectivity could have unintended side-effects; better prohibit it.
  // Besides, when can it be useful?
  assert (connectivity && !m_connectivity);

  // If the buffers manager is set and it stores a connectivity, it must match the input one
  assert (!m_buffers_manager || m_buffers_manager->get_connectivity()==connectivity);

  // Set the connectivity
  m_connectivity = connectivity;
  m_num_elems = connectivity->get_num_local_elements();
}

void BoundaryExchangeBase::
set_buffers_manager (const std::shared_ptr<MpiBuffersManager>& buffers_manager)
{
  // Functionality available only before the registration is completed
  assert (!m_registration_completed);

  // Make sure it is a valid pointer. Also, replacing the buffers manager
  // could have unintended side-effects; better prohibit it.
  // Besides, when can it be useful?
  assert (buffers_manager && !m_buffers_manager);

  // If the buffers manager stores a connectivity, and we already have one set, they must match
  assert (!buffers_manager->is_connectivity_set() || !(m_connectivity) || buffers_manager->get_connectivity()==m_connectivity);

  // Set the internal pointer
  m_buffers_manager = buffers_manager;

  // Set the connectivity in the buffers manager, if not already set
  if (!m_buffers_manager->is_connectivity_set() && m_connectivity) {
    m_buffers_manager->set_connectivity(m_connectivity);
  }

  // If I don't store a connectivity, take it from the buffers manager (if it has one)
  if (m_buffers_manager->is_connectivity_set() && !m_connectivity) {
    set_connectivity(m_buffers_manager->get_connectivity());
  }

  // Add myself as a customer of the BM
  m_buffers_manager->add_customer(this);
}

void BoundaryExchangeBase::registration_completed()
{
  // If everything is already set up, just return
  if (m_registration_completed) {
    // TODO: should we prohibit two consecutive calls of this method? It seems harmless, so I'm allowing it
    return;
  }

  // TODO: should we assert that m_registration_started=true? Or simply return if not? Can calling this
  //       method without a call to registration started be dangerous? Not sure...

  // At this point, the connectivity MUST be finalized already, and the buffers manager must be set already
  assert (m_connectivity && m_connectivity->is_finalized());
  assert (m_buffers_manager);

  // Create the MPI data types, for corners and edges
  // Note: this is the size per element, per connection. It is the number of ScalarValue's to send/receive to/from the neighbor
  // Note: for 2d/3d fields, we have 1 Real per GP (per level, in 3d). For 1d fields,
  //       we have 2 Real per level (max and min over element).

  int single_ptr_buf_size = m_num_2d_fields + m_num_3d_int_fields*NUM_LEV_P*VECTOR_SIZE;
  for (int i = 0; i < m_num_3d_fields; ++i)
    single_ptr_buf_size += m_3d_nlev_pack[i]*VECTOR_SIZE;
  m_elem_buf_size[etoi(ConnectionKind::CORNER)] = m_num_1d_fields*2*NUM_LEV*VECTOR_SIZE + single_ptr_buf_size * 1;
  m_elem_buf_size[etoi(ConnectionKind::EDGE)]   = m_num_1d_fields*2*NUM_LEV*VECTOR_SIZE + single_ptr_buf_size * NP;

  // Determine what kind of BE is this (exchange or exchange_min_max)
  m_exchange_type = m_num_1d_fields>0 ? MPI_EXCHANGE_MIN_MAX : MPI_EXCHANGE;

  // Finalize bookkeeping for any exchange on fewer than NUM_LEV levels.
  {
    bool need_nlev_pack = false;
    for (int i = 0; i < m_num_3d_fields; ++i)
      if (m_3d_nlev_pack[i] != NUM_LEV) {
        EKAT_REQUIRE_MSG(m_3d_nlev_pack[i] < NUM_LEV,
                              "Optional nlev must be <= NUM_LEV");
        EKAT_REQUIRE_MSG(m_3d_nlev_pack[i] > 0,
                              "Optional nlev must be > 0");
        need_nlev_pack = true;
        break;
      }
    if (need_nlev_pack) {
      m_3d_nlev_pack_d = ExecViewManaged<int*>("m_3d_nlev_pack_d", m_3d_nlev_pack.size());
      const auto h = Kokkos::create_mirror_view(m_3d_nlev_pack_d);
      for (int i = 0; i < m_num_3d_fields; ++i) h(i) = m_3d_nlev_pack[i];
      Kokkos::deep_copy(m_3d_nlev_pack_d, h);
    }
    // Clear the host vector if it's not needed.
    if ( ! need_nlev_pack) m_3d_nlev_pack = decltype(m_3d_nlev_pack)();
  }

  // Prohibit further registration of fields, and allow exchange
  m_registration_started   = false;
  m_registration_completed = true;

  // Optimistically build buffers here. If registration is called with largest
  // BufferManager user first, then building will occur just once, in the
  // prim_init2 call.
  build_buffer_views_and_requests();
}

void BoundaryExchangeBase::waitall()
{
  if (!m_send_pending && !m_recv_pending) {
    return;
  }

  // At this point, the connectivity MUST be valid
  assert (m_connectivity);

  // Safety check
  assert (m_buffers_manager->are_buffers_busy());

  if ( ! m_send_requests.empty())
    HOMMEXX_MPI_CHECK_ERROR(MPI_Waitall(m_send_requests.size(), m_send_requests.data(), MPI_STATUSES_IGNORE),
                            m_connectivity->get_comm().mpi_comm());
  if ( ! m_recv_requests.empty())
    HOMMEXX_MPI_CHECK_ERROR(MPI_Waitall(m_recv_requests.size(), m_recv_requests.data(), MPI_STATUSES_IGNORE),
                            m_connectivity->get_comm().mpi_comm());

  m_buffers_manager->unlock_buffers();
}

void BoundaryExchangeBase::free_requests ()
{
  auto mpi_comm = m_connectivity->get_comm().mpi_comm();
  for (auto& req : m_send_requests) {
    HOMMEXX_MPI_CHECK_ERROR(MPI_Request_free(&req),mpi_comm);
  }
  m_send_requests.clear();

  for (auto& req : m_recv_requests) {
    HOMMEXX_MPI_CHECK_ERROR(MPI_Request_free(&req),mpi_comm);
  }
  m_recv_requests.clear();
}

// A slot is the space in a communication buffer for an (element, connection)
// pair. The slot index space numbers slots so that, first, they are contiguous
// by remote PID and, second, within a PID block, each comm partner agrees on
// order of slots.
void BoundaryExchangeBase
::init_slot_idx_to_elem_conn_pair (
  std::vector<int>& slot_idx_to_elem_conn_pair,
  std::vector<int>& pids, std::vector<int>& pid_offsets)
{
  struct IP {
    size_t i, ord;
    int pid;
    bool operator< (const IP& o) const {
      if (pid < o.pid) return true;
      if (pid > o.pid) return false;
      return ord < o.ord;
    }
  };

  const auto& ucon = m_connectivity->get_h_ucon();
  const auto& ucon_ptr = m_connectivity->get_h_ucon_ptr();
  const size_t nconn = ucon.size();
  const int mce = m_connectivity->get_max_corner_elements();
  const int n_idx_per_elem = 8*mce;
  std::vector<IP> i2remote(nconn);

  for (size_t k = 0; k < nconn; ++k) {
    const auto& info = ucon(k);
    auto& i2r = i2remote[k];
    // Original sequence through connections.
    i2r.i = k;
    // An ordering of the message buffer upon which both members of the
    // communication pair agree.
    const auto& lgp = info.local.gid < info.remote.gid ? info.local : info.remote;
    i2r.ord = lgp.gid*n_idx_per_elem + lgp.dir*mce + lgp.dir_idx;
    // If local, indicate with -1, which is < the smallest pid of 0.
    i2r.pid = -1;
    if (info.sharing == etoi(ConnectionSharing::SHARED))
      i2r.pid = info.remote_pid;
  }

  // Sort so that, first, all connections having the same remote_pid are
  // contiguous; second, within such a block, ord is ascending. The first lets
  // us set up comm buffers so monolithic messages slot right in. The second
  // means that the send and recv partners agree on how a monolithic message is
  // packed.
  std::sort(i2remote.begin(), i2remote.end());

  // Collect the unique remote_pids and get the offsets of the contiguous blocks
  // of them.
  slot_idx_to_elem_conn_pair.resize(nconn);
  pids.clear();
  pid_offsets.clear();
  int prev_pid = -2;
  for (size_t k = 0; k < nconn; ++k) {
    const auto& i2r = i2remote[k];
    if (i2r.pid > prev_pid && i2r.pid != -1) {
      pids.push_back(i2r.pid);
      pid_offsets.push_back(k);
      prev_pid = i2r.pid;
    }
    slot_idx_to_elem_conn_pair[k] = i2r.i;
  }
  pid_offsets.push_back(nconn);
}


} // namespace Homme
