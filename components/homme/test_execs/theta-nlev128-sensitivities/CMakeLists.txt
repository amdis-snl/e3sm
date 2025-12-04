# We run the theta-nlev128-kokkos executable for 1 day for different values
# of a parameter(e.g., p=val,p=val+0.1,p=val+0.01,p=val+0.001),
# and compute sensitivities via finite differences(FD).
# Then, we compare the FD sensitivity with what we obtain via Sacado for p=val.
# We run the model for 1h, to keep the test relatively fast

# Create the executable. Note, we cannot recylce the one from ../theta-nlev128-kokkos,
# since we need to set a CPP macro that triggers the hyperviscosity factor nu to be
# initialized as a Fad type (with deriv(0)=1)
THETAL_KOKKOS_SETUP()
createTestExec(theta-nlev128-sens theta-l_kokkos  4  4  128   ${PREQX_USE_PIO}   FALSE        10)
target_compile_definitions(theta-nlev128-sens PUBLIC HOMMEXX_TEST_HYPERVIS_FAD)

set(TEST_NAME theta-nlev128-sensitivities)
if(USE_NUM_PROCS)
  if(USE_MPI_RUN_SCRIPT)
    set(EXEC_LAUNCHER "${USE_MPI_RUN_SCRIPT} ${USE_NUM_PROCS} ${CMAKE_CURRENT_BINARY_DIR}/theta-nlev128-sens")
  else()
    set(EXEC_LAUNCHER "${USE_MPIEXEC} -n ${USE_NUM_PROCS} ${CMAKE_CURRENT_BINARY_DIR}/theta-nlev128-sens")
  endif()
else()
  set(EXEC_LAUNCHER "${CMAKE_CURRENT_BINARY_DIR}/theta-nlev128-sens")
endif()

file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/vcoord)
file(COPY ${HOMME_SOURCE_DIR}/test/vcoord/sabm-128.ascii DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/vcoord)
file(COPY ${HOMME_SOURCE_DIR}/test/vcoord/sabi-128.ascii DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/vcoord)
file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/output)

#######################################
#             Base value              #
#######################################

set(OUTPUT_PREFIX base_)
set(NU_PERTURB 0)
set(SENS_FREQ 1)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/namelist.nl
               ${CMAKE_CURRENT_BINARY_DIR}/namelist.nl)

add_test(NAME ${TEST_NAME}-base
  COMMAND ${CMAKE_COMMAND} -E env bash -c "${EXEC_LAUNCHER} < namelist.nl")
set_tests_properties(${TEST_NAME}-base PROPERTIES
  LABELS AMDIS
  FIXTURES_SETUP base_output)
set(fixtures base_output)

#######################################
#           Perturb values            #
#######################################

# This ensures we don't save sensitivities, as the perturbed runs are just to compute the perturbed state
set(SENS_FREQ 0)

set(PERTURBATIONS 1e0 1e-2 1e-4 1e-6)
foreach(p IN ITEMS ${PERTURBATIONS})
  set(OUTPUT_PREFIX perturbed_${p}_)
  set(NU_PERTURB ${p})
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/namelist.nl
                 ${CMAKE_CURRENT_BINARY_DIR}/namelist_${p}.nl)

  add_test(NAME ${TEST_NAME}-pert-${p}
    COMMAND ${CMAKE_COMMAND} -E env bash -c "${EXEC_LAUNCHER} < namelist_${p}.nl")
  set_tests_properties(${TEST_NAME}-pert-${p} PROPERTIES
    LABELS AMDIS
    FIXTURES_SETUP pert-${p})
  list(APPEND fixtures pert-${p})
endforeach()

#######################################
#    Compute FD and compare values    #
#######################################

add_test(NAME ${TEST_NAME}-check
  COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/sens_check.py
    -v u v Th
    -d ${PERTURBATIONS}
    -p perturbed
    -b base
    -t jw_baroclinic
    --tol 1e-3
  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/output)

message("fixtures: ${fixtures}")
set_tests_properties(${TEST_NAME}-check PROPERTIES
  LABELS AMDIS
  FIXTURES_REQUIRED "${fixtures}")
