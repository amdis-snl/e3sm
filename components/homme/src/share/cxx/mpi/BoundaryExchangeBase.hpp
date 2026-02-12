/********************************************************************************
 * HOMMEXX 1.0: Copyright of Sandia Corporation
 * This software is released under the BSD license
 * See the file 'COPYRIGHT' in the HOMMEXX/src/share/cxx directory
 *******************************************************************************/

#ifndef HOMMEXX_BOUNDARY_EXCHANGE_BASE_HPP
#define HOMMEXX_BOUNDARY_EXCHANGE_BASE_HPP

#include "Types.hpp" // For ExecViewManaged

#include <mpi.h>

#include <memory>
#include <vector>
#include <string>

namespace Homme
{

// Forward declaration
class MpiBuffersManager;
class Connectivity;

/*
 * BoundaryExchangeBase: a base class for BoundaryExchange objects
 *
 * This class (BE) takes care of exchanging the values of one or more fields in the
 * GP on the boundary of the elements with the neighboring elements. In particular,
 * it takes care of packing the values into some buffers, performing the exchange
 * (usually, this includes some MPI calls, unless running in serial mode),
 * and then unpacking the values, accumulating them into the receiving elements.
 * This process can be done for an arbitrary number of 2d fields (that is,
 * no vertical levels) and 3d fields (with vertical levels). If you have a
 * vector field of dimension DIM, you need to register DIM separate scalar
 * fields. Internally, for each input field the BE object stores a bunch of
 * separate Views, that view the input field at each element and component
 * (if vector field). When the exchange method is called, ALL the stored
 * fields are packed/exchanged/unpacked. Therefore, if you have two sets
 * of fields that need to be exchanged at different times, you need to
 * register them into two separate BE objects.
 *
 * The registration happens in three steps:
 *
 *  - a call to set_num_fields, which sets the number of 1d, 2d and 3d fields
 *    that will be exchanged. Once this method is called, it cannot be
 *    called again, unless the method clean_up is called first.
 *  - a number of calls to one of more of the register_field(...), methods,
 *    which set the fields into the BE class. You cannot register more fields
 *    than declared in the set_num_fields call. However you can, if you want,
 *    register less fields, although this scenario is not tested, and may
 *    be buggy, so you are probably better off calling set_num_fields with
 *    the actual number of fields you are going to register. Note that you
 *    are not allowed to call register_field(...) before set_num_fields.
 *    NOTE: you are NOT allowed to register 1d fields together with
 *          2d/3d fields. The idea is that 1d fields are exchanged only as
 *          min/max quantities (so they are not accumulated). Please, use
 *          two different BE objects for accumulation and for min/max.
 *  - a call to registration_completed, which ends the registration phase,
 *    and sets up all the internal structure to prepare for calls to
 *    exchange(). This method MUST be called BEFORE any call to exchange.
 *
 * This class relies on the MpiBuffersManager (BM) class for the handling of the buffers.
 * See MpiBuffersManager header for more info on that. As explained above,
 * you may have different BE objects, which are however never used at the
 * same time. Therefore, it makes sense to reuse the same BM for all of them.
 * For this reason, the BM can serve multiple 'customers'. The BE and BM
 * classes are linked by a provider-customer relationship. It is up to the BE
 * to register itself as a customer in the stored BM, and to unregister
 * itself before going out of scope, to make sure the BM does not keep
 * a dangling pointer to a non-existent customer. This is why BE's destructor
 * automatically removes the 'this' object from the stored BM's customers list
 * (assuming there is a stored BM, otherwise nothing happens).
 *
 * In order to work correctly, BE needs a valid Connectivity and a valid
 * BM (both stored as shared_ptr). They can be set at construction time
 * or later, via a setter method. There are only a few rules:
 *
 *  - once one is set, you cannot reset it
 *  - the Connectivity must be set BEFORE any call to set_num_fields
 *  - the BM must be set BEFORE any call to registration_completed
 *
 */

class BoundaryExchangeBase
{
public:

  ~BoundaryExchangeBase();

  // Set the connectivity if default constructor was used
  void set_connectivity (const std::shared_ptr<Connectivity>& connectivity);

  // Set the buffers manager (registration must not be completed)
  void set_buffers_manager (const std::shared_ptr<MpiBuffersManager>& buffers_manager);

  size_t get_scalar_size () const { return m_scalar_size; }

  // If you are really not sure whether we are still transmitting, you can make sure we're done by calling this
  void waitall ();

  // Public members. We used to have set/get method, but that's just like having the member public...
  std::string m_label;          // Optional label to help parsing debug output
  int m_diagnostics_level = 0;  // Whether to print diag output during exchange (0: none, 1: post, 2: pre-and-post)

  // Check whether fields registration has already started/finished
  bool is_registration_started   () const { return m_registration_started;   }
  bool is_registration_completed () const { return m_registration_completed; }

  // Get the number of 1d/2d/3d/3d_int fields that this object handles
  int get_num_1d_fields     () const { return m_num_1d_fields; }
  int get_num_2d_fields     () const { return m_num_2d_fields; }
  int get_num_3d_fields     () const { return m_num_3d_fields; }
  int get_num_3d_int_fields () const { return m_num_3d_int_fields; }

  // Size the buffers, and initialize the MPI types
  void registration_completed();

protected:

  // Make MpiBuffersManager a friend, so it can call the method underneath
  friend class MpiBuffersManager;
  virtual void clear_buffer_views_and_requests () = 0;

  // Protected, so we only instantiate derived types
  BoundaryExchangeBase() = default;

  virtual void build_buffer_views_and_requests () = 0;

  void init_slot_idx_to_elem_conn_pair(
    std::vector<int>& h_slot_idx_to_elem_conn_pair,
    std::vector<int>& pids, std::vector<int>& pids_os);
  void free_requests();

  short int m_exchange_type;

  std::shared_ptr<Connectivity>   m_connectivity;

  MPI_Datatype              m_scalar_dtype;
  size_t                    m_scalar_size;
  int                       m_elem_buf_size[2];

  std::vector<MPI_Request>  m_send_requests;
  std::vector<MPI_Request>  m_recv_requests;

  // This class contains all the buffers to be stuffed in the buffers views, and used in pack/unpack,
  // as well as the mpi buffers used in MPI calls (which are the same as the former if MPIMemSpace=ExecMemSpace),
  // and the blackhole buffers (used for missing connections)
  std::shared_ptr<MpiBuffersManager> m_buffers_manager;

  std::vector<int> m_3d_nlev_pack;        // during registration
  ExecViewManaged<int*> m_3d_nlev_pack_d; //  after registration

  // The number of registered fields
  int         m_num_1d_fields = 0;    // Without counting the 2x factor due to min/max fields
  int         m_num_2d_fields = 0;
  int         m_num_3d_fields = 0;
  int         m_num_3d_int_fields = 0;

  // The following flags are used to ensure that a bad user does not call setup/cleanup/registration
  // methods of this class in an order that generate errors. And if he/she does, we try to avoid errors.
  bool        m_registration_started = false;
  bool        m_registration_completed = false;
  bool        m_buffer_views_and_requests_built = false;
  bool        m_cleaned_up = true;
  bool        m_send_pending = false;
  bool        m_recv_pending = false;

  int         m_num_elems = -1;

};

} // namespace Homme

#endif // HOMMEXX_BOUNDARY_EXCHANGE_BASE_HPP
