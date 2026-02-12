/********************************************************************************
 * HOMMEXX 1.0: Copyright of Sandia Corporation
 * This software is released under the BSD license
 * See the file 'COPYRIGHT' in the HOMMEXX/src/share/cxx directory
 *******************************************************************************/

#include "MpiBuffersManager.hpp"

#include "BoundaryExchangeBase.hpp"
#include "Connectivity.hpp"

namespace Homme
{

MpiBuffersManager::MpiBuffersManager (std::shared_ptr<Connectivity> connectivity)
{
  set_connectivity(connectivity);
}

MpiBuffersManager::~MpiBuffersManager ()
{
  // Check that all the customers un-registered themselves
  assert (m_num_customers==0);

  // Check our buffers are not busy
  assert (!m_buffers_busy);
}

void MpiBuffersManager::check_for_reallocation ()
{
  for (const auto& it : m_customers) {
    update_requested_sizes (it.first);
  }
}

void MpiBuffersManager::set_connectivity (std::shared_ptr<Connectivity> connectivity)
{
  // We don't allow a null connectivity, or a change of connectivity
  assert (connectivity && !m_connectivity);

  m_connectivity = connectivity;
}

void MpiBuffersManager::allocate_buffers ()
{
  // If views are marked as valid, they are already allocated, and no other
  // customer has requested a larger size
  if (m_views_are_valid) {
    return;
  }

  // The buffers used for packing/unpacking
  m_send_buffer  = ExecViewManaged<char*>("send buffer",  m_mpi_buffer_size);
  m_recv_buffer  = ExecViewManaged<char*>("recv buffer",  m_mpi_buffer_size);
  m_local_buffer = ExecViewManaged<char*>("local buffer", m_local_buffer_size);
  m_blackhole_send_buffer = ExecViewManaged<char*>("blackhole array",m_blackhole_buffer_size);
  m_blackhole_recv_buffer = ExecViewManaged<char*>("blackhole array",m_blackhole_buffer_size);
  Kokkos::deep_copy(m_blackhole_send_buffer,0.0);
  Kokkos::deep_copy(m_blackhole_recv_buffer,0.0);

  // The buffers used in MPI calls
  m_mpi_send_buffer = Kokkos::create_mirror_view(decltype(m_mpi_send_buffer)::execution_space(),m_send_buffer);
  m_mpi_recv_buffer = Kokkos::create_mirror_view(decltype(m_mpi_recv_buffer)::execution_space(),m_recv_buffer);

  m_views_are_valid = true;

  // Tell to all our customers that they need to redo the setup of the internal buffer views
  for (auto& be_ptr : m_customers) {
    // Invalidate buffer views and requests in the customer (if none built yet, it's a no-op)
    be_ptr.first->clear_buffer_views_and_requests ();
  }
}

void MpiBuffersManager::lock_buffers ()
{
  // Make sure we are not trying to lock buffers already locked
  assert (!m_buffers_busy);

  m_buffers_busy = true;
}

void MpiBuffersManager::unlock_buffers ()
{
  // TODO: I am not checking if the buffers are locked. This allows to call
  //       the method twice in a row safely. Is this a bad idea?
  m_buffers_busy = false;
}

char* MpiBuffersManager::get_send_buffer () const
{
  // We ensure that the buffers are valid
  assert(m_views_are_valid);
  return m_send_buffer.data();
}

char* MpiBuffersManager::get_recv_buffer () const
{
  // We ensure that the buffers are valid
  assert(m_views_are_valid);
  return m_recv_buffer.data();
}

char* MpiBuffersManager::get_local_buffer () const
{
  // We ensure that the buffers are valid
  assert(m_views_are_valid);
  return m_local_buffer.data();
}

char* MpiBuffersManager::get_mpi_send_buffer() const
{
  // We ensure that the buffers are valid
  assert(m_views_are_valid);
  return m_mpi_send_buffer.data();
}

char* MpiBuffersManager::get_mpi_recv_buffer() const
{
  // We ensure that the buffers are valid
  assert(m_views_are_valid);
  return m_mpi_recv_buffer.data();
}

char* MpiBuffersManager::get_blackhole_send_buffer () const
{
  // We ensure that the buffers are valid
  assert(m_views_are_valid);
  return m_blackhole_send_buffer.data();
}

char* MpiBuffersManager::get_blackhole_recv_buffer () const
{
  // We ensure that the buffers are valid
  assert(m_views_are_valid);
  return m_blackhole_recv_buffer.data();
}

void MpiBuffersManager::add_customer (BoundaryExchangeBase* add_me)
{
  // We don't allow null customers (although this should never happen)
  assert (add_me!=nullptr);

  // We also don't allow re-registration
  assert (m_customers.find(add_me)==m_customers.end());

  // Add to the list of customers
  m_customers.emplace(add_me,CustomerNeeds{0,0});

  // Update the number of customers
  ++m_num_customers;

  // If this customer has already started the registration, we can already update the buffers sizes
  if (add_me->is_registration_started()) {
    update_requested_sizes(add_me);
  }
}

void MpiBuffersManager::remove_customer (BoundaryExchangeBase* remove_me)
{
  // We don't allow null customers (although this should never happen)
  assert (remove_me!=nullptr);

  // Perhaps overzealous, but won't hurt: we should have customers
  assert (m_num_customers>0);

  // Find the customer
  auto it = m_customers.find(remove_me);

  // We don't allow removal of non-customers
  assert (it!=m_customers.end());

  // Remove the customer and its needs
  m_customers.erase(it);

  // Decrease number of customers
  --m_num_customers;
}

void MpiBuffersManager::update_requested_sizes (BoundaryExchangeBase* customer)
{
  // Make sure connectivity is valid
  assert (m_connectivity && m_connectivity->is_finalized());

  // Make sure this is a customer
  assert (m_customers.find(customer)!=m_customers.end());

  // Get the number of fields that this customer has
  const int num_1d_fields = customer->get_num_1d_fields();
  const int num_2d_fields = customer->get_num_2d_fields();
  const int num_3d_fields = customer->get_num_3d_fields();
  const int num_3d_int_fields = customer->get_num_3d_int_fields();
  const size_t scalar_size = customer->get_scalar_size();

  // Compute the requested buffers sizes and compare with stored ones
  auto& needs = m_customers.at(customer);
  required_buffer_sizes (scalar_size, num_1d_fields, num_2d_fields, num_3d_fields, num_3d_int_fields,
                         needs.mpi_buffer_size, needs.local_buffer_size, needs.blackhole_buffer_size);
  if (needs.mpi_buffer_size>m_mpi_buffer_size) {
    // Update the current mpi buf size
    m_mpi_buffer_size = needs.mpi_buffer_size;

    // Mark the views as invalid
    m_views_are_valid = false;
  }

  if(needs.local_buffer_size>m_local_buffer_size) {
    // Update the current local buf size
    m_local_buffer_size = needs.local_buffer_size;

    // Mark the views as invalid
    m_views_are_valid = false;
  }

  if(needs.blackhole_buffer_size>m_blackhole_buffer_size) {
    // Update the current blackhole buf size
    m_blackhole_buffer_size = needs.blackhole_buffer_size;

    // Mark the views as invalid
    m_views_are_valid = false;
  }
}

void MpiBuffersManager::required_buffer_sizes (const size_t scalar_size,
                                               const int num_1d_fields, const int num_2d_fields,
                                               const int num_3d_fields, const int num_3d_interface_fields,
                                               size_t& mpi_buffer_size, size_t& local_buffer_size, size_t& blackhole_buffer_size) const
{
  mpi_buffer_size = local_buffer_size = blackhole_buffer_size = 0;

  // The buffer size for each connection kind
  // Note: for 2d/3d fields, we have 1 Real per GP (per level, in 3d). For 1d fields,
  //       we have 2 Real per level (max and min over element).
  int elem_buf_size[2];
  const int pt_buf_size = num_2d_fields + num_3d_fields*NUM_LEV*VECTOR_SIZE + num_3d_interface_fields*NUM_LEV_P*VECTOR_SIZE;
  elem_buf_size[etoi(ConnectionKind::CORNER)] = num_1d_fields*2*NUM_LEV*VECTOR_SIZE + pt_buf_size * 1;
  elem_buf_size[etoi(ConnectionKind::EDGE)]   = num_1d_fields*2*NUM_LEV*VECTOR_SIZE + pt_buf_size * NP;

  // Compute the requested buffers sizes and compare with stored ones
  mpi_buffer_size += elem_buf_size[etoi(ConnectionKind::CORNER)] * m_connectivity->get_num_connections<HostMemSpace>(ConnectionSharing::SHARED,ConnectionKind::CORNER);
  mpi_buffer_size += elem_buf_size[etoi(ConnectionKind::EDGE)]   * m_connectivity->get_num_connections<HostMemSpace>(ConnectionSharing::SHARED,ConnectionKind::EDGE);

  local_buffer_size += elem_buf_size[etoi(ConnectionKind::CORNER)] * m_connectivity->get_num_connections<HostMemSpace>(ConnectionSharing::LOCAL,ConnectionKind::CORNER);
  local_buffer_size += elem_buf_size[etoi(ConnectionKind::EDGE)]   * m_connectivity->get_num_connections<HostMemSpace>(ConnectionSharing::LOCAL,ConnectionKind::EDGE);

  local_buffer_size *= scalar_size;
  mpi_buffer_size *= scalar_size;

  // The "fake" buffer used for MISSING connections has a fixed size (in terms of scalars)
  blackhole_buffer_size = scalar_size * 2 * NUM_LEV * VECTOR_SIZE;
}

} // namespace Homme
