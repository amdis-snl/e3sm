/********************************************************************************
 * HOMMEXX 1.0: Copyright of Sandia Corporation
 * This software is released under the BSD license
 * See the file 'COPYRIGHT' in the HOMMEXX/src/share/cxx directory
 *******************************************************************************/

#ifndef HOMMEXX_CONTEXT_HPP
#define HOMMEXX_CONTEXT_HPP

#include <any>
#include <string>
#include <map>
#include <memory>
#include <functional>

#include <ekat_assert.hpp>

namespace Homme {

/* A Context manages resources that are morally singletons. Context is
 * meant to have two roles. First, a Context singleton is the only singleton in
 * the program. Second, a context need not be a singleton, and each Context
 * object can have different Elements, ReferenceElement, etc., objects. (That
 * probably isn't needed, but Context immediately supports it.)
 *
 * Finally, Context has two singleton functions: singleton(), which returns
 * Context&, and finalize_singleton(). The second is called in a unit test exe
 * main before Kokkos::finalize().
 */
class Context {
public:

  // Getters for a managed object.
  bool has (const std::string& key) const {
    return m_members.find(key)!=m_members.end();
  }

  // Setters for a managed object.
  template<typename ConcreteType, typename... Args>
  ConcreteType& create (const std::string& key, Args&&... args);

  template<typename ConcreteType, typename... Args>
  ConcreteType& create (Args&&... args) {
    return create<ConcreteType>(std::string(typeid(ConcreteType).name()),args...);
  }

  template<typename ConcreteType>
  void create_ref (const std::string& key, ConcreteType& src);

  // More relaxed than create, it won't throw if the
  // object already exists, and simply return it.
  // NOTE: this is allows more flexibility in the cxx-f90
  //       interfaces, in that you have more freedom in
  //       the order of certain function calls.
  //       With 'create', you have to either check if an
  //       object was already created (and create it if it wasn't),
  //       or make sure that the function that calls 'create<T>'
  //       is *always* invoked *before* those that call 'get<T>'.
  template<typename ConcreteType, typename... Args>
  ConcreteType& create_if_not_there (const std::string& key, Args&&... args);

  // Getters for a managed object.
  template<typename ConcreteType>
  ConcreteType& get (const std::string& key);

  std::any& get (const std::string& key);

  template<typename ConcreteType>
  std::shared_ptr<ConcreteType> get_ptr (const std::string& key) const;

  // Exactly one singleton.
  static Context& singleton() {
    static Context c;
    return c;
  }

  static void finalize_singleton() {
    singleton().clear();
  }
private:

  std::map<std::string,std::any> m_members;
  std::map<std::string, bool>    m_is_ref_wrapper;

  // Clear the objects Context manages.
  void clear() {
    m_members.clear();
    m_is_ref_wrapper.clear();
  }
};

// ==================== IMPLEMENTATION =================== //

template<typename ConcreteType, typename... Args>
ConcreteType& Context::create_if_not_there (const std::string& key, Args&&... args)
{
  auto it = m_members.find(key);
  if (it==m_members.end()) {
    auto ptr = std::make_shared<ConcreteType>(std::forward<Args>(args)...);
    m_members[key] = ptr;
    m_is_ref_wrapper[key] = false;
    return *ptr;
  }
  auto ptr = *std::any_cast<std::shared_ptr<ConcreteType>>(&it->second);
  return *ptr;
}

template<typename ConcreteType, typename... Args>
ConcreteType& Context::create (const std::string& key, Args&&... args)
{
  EKAT_REQUIRE_MSG(!has(key),
      "Error! An object for the concrete type " + key +
      " is already stored. The 'Context' class does not allow overwriting or duplicates.\n");

  auto ptr = std::make_shared<ConcreteType>(std::forward<Args>(args)...);
  m_members[key] = ptr;
  m_is_ref_wrapper[key] = false;

  return *ptr;
}

template<typename ConcreteType>
void Context::create_ref (const std::string& key, ConcreteType& src)
{
  EKAT_REQUIRE_MSG(!has(key),
      "Error! An object for the concrete type " + key +
      " is already stored. The 'Context' class does not allow overwriting or duplicates.\n");

  m_members[key] = std::reference_wrapper<ConcreteType>(src);
  m_is_ref_wrapper[key] = true;
}

template<typename ConcreteType>
ConcreteType& Context::get (const std::string& key)
{
  auto& any = this->get(key);
  if (m_is_ref_wrapper.at(key)) {
    return std::any_cast<std::reference_wrapper<ConcreteType>>(any).get();
  } else {
    return *std::any_cast<std::shared_ptr<ConcreteType>>(any);
  }
}

std::any& Context::get (const std::string& key)
{
  auto it = m_members.find(key);
  EKAT_REQUIRE_MSG(it!=m_members.end(), "Error! Context member '" + key + "' not found.\n");
  return it->second;
}

template<typename ConcreteType>
std::shared_ptr<ConcreteType> Context::get_ptr(const std::string& key) const
{
  auto it = m_members.find(key);
  EKAT_REQUIRE_MSG(it!=m_members.end(), "Error! Context member '" + key + "' not found.\n");
  EKAT_REQUIRE_MSG(!m_is_ref_wrapper.at(key),
      "Error! Context member '" + key + "' is only available as a reference.\n");

  return std::any_cast<std::shared_ptr<ConcreteType>>(it->second);
}

} // namespace Homme

#endif // HOMMEXX_CONTEXT_HPP
