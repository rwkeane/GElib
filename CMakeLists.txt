cmake_minimum_required(VERSION 3.18)
message(STATUS "Initial CMAKE_CURRENT_BINARY_DIR is ${CMAKE_CURRENT_BINARY_DIR}")
message(STATUS "Initial CMAKE_SOURCE_DIR is ${CMAKE_SOURCE_DIR}")
project(gelib LANGUAGES CXX)
enable_testing()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON) # Useful for debugging and IDEs
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE) # For RPATH handling

# Placeholder for future CUDA integration
option(GELIB_WITH_CUDA "Enable CUDA support (manual integration needed)" OFF)

# --- Add further CMake configurations below ---
# Add cnine submodule
set(WITH_CUDA OFF CACHE BOOL "Disable CUDA in cnine" FORCE)
set(BUILD_PYTHON OFF CACHE BOOL "Disable Python in cnine" FORCE)
add_subdirectory(deps/cnine)

# --- Core C++ Library (gelib_cpp_core) ---

# GElib's core logic for the Python bindings is primarily header-based (.hpp files)
# or compiled directly into the Python extension via GElib_py.cpp (which includes GElib_base.cpp).
# Therefore, gelib_cpp_core is an INTERFACE library to manage include paths,
# compiler definitions, and transitive dependencies like cnine and PyTorch.
add_library(gelib_cpp_core INTERFACE)

# 3. Set C++17 standard for the target
target_compile_features(gelib_cpp_core INTERFACE cxx_std_17) # For INTERFACE libraries, features are INTERFACE

# 4. Include Directories
target_include_directories(gelib_cpp_core INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>  # Project's public headers
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/core>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/SO3>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/O3>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/cuda>   # For potential .hpp files like cuda/utils.hpp
    # cnine includes are typically handled by linking with cnine's target if it uses INTERFACE_INCLUDE_DIRECTORIES
    # If cnine target exists and has INTERFACE_INCLUDE_DIRECTORIES, this will be inherited via target_link_libraries
    # However, explicitly adding cnine's main include dir if necessary:
    # $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/deps/cnine/include>
    # It's better to rely on cnine's CMake target definition.
)
# Link cnine and inherit its interface properties (like include directories)
# Assuming 'cnine' is the target name defined in deps/cnine/CMakeLists.txt
# If cnine itself is a library target:
if(TARGET cnine) # Or the actual target name for the cnine library
    # Also add pybind11::headers here to ensure cnine and gelib C++ code see pybind11 headers
    # if any template/inline code needs them, in a context compatible with torch/extension.h
    target_link_libraries(gelib_cpp_core INTERFACE cnine pybind11::headers)
    message(STATUS "gelib_cpp_core linked with cnine and pybind11::headers.")
else()
    target_link_libraries(gelib_cpp_core INTERFACE pybind11::headers) # Still add pybind11 headers
    message(WARNING "cnine target not found. Include directories from cnine might be missing for gelib_cpp_core.")
endif()


# 5. Compiler Flags and Definitions (from setup.py _cxx_compile_args)
set(GELIB_CXX_COMPILE_FLAGS
    -Wno-sign-compare
    -Wno-deprecated-declarations
    -Wno-unused-variable
    -Wno-reorder
    -Wno-reorder-ctor
    -Wno-overloaded-virtual
    # Add flags from setup.py's 'copy_warnings' if they were true by default or desired
    # -DCNINE_COPY_WARNINGS
    # -DCNINE_ASSIGN_WARNINGS
    # -DCNINE_MOVE_WARNINGS
    # -DCNINE_MOVEASSIGN_WARNINGS
    # -DGELIB_COPY_WARNINGS
    # -DGELIB_MOVE_WARNINGS
    # -DGELIB_CONVERT_WARNINGS
    # Add flags from setup.py's 'torch_convert_warnings' if they were true by default or desired
    # -DCNINE_ATEN_CONVERT_WARNINGS # This is a definition, not a compile option
)
# Add PyTorch CXX flags which often include ABI settings
if(TORCH_CXX_FLAGS)
  target_compile_options(gelib_cpp_core INTERFACE ${TORCH_CXX_FLAGS})
  message(STATUS "Added TORCH_CXX_FLAGS to gelib_cpp_core.")
endif()
target_compile_options(gelib_cpp_core INTERFACE ${GELIB_CXX_COMPILE_FLAGS})


set(GELIB_CXX_DEFINITIONS
    _WITH_ATEN # PyTorch related
    CNINE_RANGE_CHECKING
    CNINE_SIZE_CHECKING
    CNINE_DEVICE_CHECKING
    GELIB_RANGE_CHECKING
    WITH_FAKE_GRAD
)
# From setup.py's 'copy_warnings' and 'torch_convert_warnings' if enabled:
list(APPEND GELIB_CXX_DEFINITIONS
    CNINE_COPY_WARNINGS
    CNINE_ASSIGN_WARNINGS
    CNINE_MOVE_WARNINGS
    CNINE_MOVEASSIGN_WARNINGS
    GELIB_COPY_WARNINGS
    GELIB_MOVE_WARNINGS
    GELIB_CONVERT_WARNINGS
    CNINE_ATEN_CONVERT_WARNINGS
)
target_compile_definitions(gelib_cpp_core INTERFACE ${GELIB_CXX_DEFINITIONS}) # INTERFACE because some might affect headers

# Conditional CUDA-context C++ definitions (will be OFF for this task)
if(GELIB_WITH_CUDA)
    target_compile_definitions(gelib_cpp_core INTERFACE _WITH_CUDA _WITH_CUBLAS) # INTERFACE
    # Add any other C++ definitions that were conditional on CUDA in setup.py
endif()

# --- PyTorch Integration ---
# This relies on CMAKE_PREFIX_PATH being set correctly to find PyTorch
# in the current Python environment. scikit-build-core should handle this,
# or it can be set manually:
# e.g., cmake -DCMAKE_PREFIX_PATH=$(python -c 'import torch; print(torch.utils.cmake_prefix_path)') ...
find_package(Torch 2.7.0 REQUIRED)

# Check if Torch was found and print a message
if(TORCH_FOUND)
    message(STATUS "PyTorch found. Version: ${TORCH_VERSION}")
    # message(STATUS "PyTorch include directories: ${TORCH_INCLUDE_DIRS}") # For debugging if needed
    # message(STATUS "PyTorch libraries: ${TORCH_LIBRARIES}") # For debugging if needed
else()
    message(FATAL_ERROR "PyTorch was not found. Ensure PyTorch is installed in your Python environment and CMAKE_PREFIX_PATH is set correctly if building manually.")
endif()

# Link gelib_cpp_core with PyTorch
# The _WITH_ATEN definition is already added to gelib_cpp_core's definitions.
if(TARGET Torch::torch_cpu AND TARGET Torch::torch_global_deps)
    target_link_libraries(gelib_cpp_core INTERFACE Torch::torch_cpu Torch::torch_global_deps)
    message(STATUS "Linking gelib_cpp_core with Torch::torch_cpu and Torch::torch_global_deps.")
elseif(TORCH_LIBRARIES)
    # Fallback if specific Torch:: targets are not available (older PyTorch versions might use TORCH_LIBRARIES)
    target_link_libraries(gelib_cpp_core INTERFACE ${TORCH_LIBRARIES})
    message(STATUS "Linking gelib_cpp_core with PyTorch libraries: ${TORCH_LIBRARIES}.")
else()
    message(WARNING "Could not determine PyTorch libraries to link against gelib_cpp_core. Check PyTorch CMake configuration.")
endif()

# Also, ensure PyTorch include directories are available to gelib_cpp_core
# This is often handled by target_link_libraries with modern CMake targets like Torch::torch_cpu
# but can be added explicitly if needed for older PyTorch versions or unusual setups.
if(TORCH_INCLUDE_DIRS)
    target_include_directories(gelib_cpp_core INTERFACE ${TORCH_INCLUDE_DIRS})
endif()

# --- pybind11 Integration ---
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
set(PYBIND11_FINDPYTHON ON)
find_package(pybind11 REQUIRED)

if(NOT Python3_FOUND)
    message(FATAL_ERROR "Python3 development components were not found.")
endif()
if(NOT pybind11_FOUND)
    message(FATAL_ERROR "pybind11 was not found.")
endif()

# --- Manually Find PyTorch Libraries for Linking and RPATH ---
set(PYTORCH_LIB_DIR "")
if(TORCH_FOUND AND PYTHON_EXECUTABLE)
    execute_process(
        COMMAND "${PYTHON_EXECUTABLE}" -c "import os; import torch; print(os.path.dirname(torch.__file__))"
        OUTPUT_VARIABLE PYTORCH_PACKAGE_DIR
        OUTPUT_STRIP_TRAILING_WHITESPACE
        RESULT_VARIABLE PYTORCH_LOCATE_SCRIPT_RESULT
    )
    if(PYTORCH_LOCATE_SCRIPT_RESULT EQUAL 0 AND PYTORCH_PACKAGE_DIR AND EXISTS "${PYTORCH_PACKAGE_DIR}/lib")
        set(PYTORCH_LIB_DIR_TEMP "${PYTORCH_PACKAGE_DIR}/lib")
        get_filename_component(PYTORCH_LIB_DIR ${PYTORCH_LIB_DIR_TEMP} ABSOLUTE) # Normalize path
        message(STATUS "Found PyTorch package lib directory via Python script: ${PYTORCH_LIB_DIR}")
    elseif(TORCH_CMAKE_PREFIX_PATH) # Fallback to TORCH_CMAKE_PREFIX_PATH if Python script fails or path doesn't exist
        if(EXISTS "${TORCH_CMAKE_PREFIX_PATH}/lib")
            set(PYTORCH_LIB_DIR_TEMP "${TORCH_CMAKE_PREFIX_PATH}/lib")
            get_filename_component(PYTORCH_LIB_DIR ${PYTORCH_LIB_DIR_TEMP} ABSOLUTE)
            message(STATUS "Found PyTorch package lib directory via TORCH_CMAKE_PREFIX_PATH/lib: ${PYTORCH_LIB_DIR}")
        elseif(EXISTS "${TORCH_CMAKE_PREFIX_PATH}/../../lib") # Common for pip install structure
             set(PYTORCH_LIB_DIR_TEMP "${TORCH_CMAKE_PREFIX_PATH}/../../lib")
             get_filename_component(PYTORCH_LIB_DIR ${PYTORCH_LIB_DIR_TEMP} ABSOLUTE)
             message(STATUS "Found PyTorch package lib directory via TORCH_CMAKE_PREFIX_PATH/../../lib: ${PYTORCH_LIB_DIR}")
        endif()
    endif()

    if(PYTORCH_LIB_DIR AND EXISTS "${PYTORCH_LIB_DIR}")
        find_library(TORCH_PYTHON_LIBRARY_PATH NAMES torch_python HINTS "${PYTORCH_LIB_DIR}" NO_DEFAULT_PATH)
        find_library(TORCH_LIBRARY_PATH NAMES torch HINTS "${PYTORCH_LIB_DIR}" NO_DEFAULT_PATH) # General libtorch.so
        find_library(TORCH_CPU_LIBRARY_PATH NAMES torch_cpu HINTS "${PYTORCH_LIB_DIR}" NO_DEFAULT_PATH) # Specific CPU lib
        find_library(C10_LIBRARY_PATH NAMES c10 HINTS "${PYTORCH_LIB_DIR}" NO_DEFAULT_PATH)

        if(NOT TORCH_PYTHON_LIBRARY_PATH)
            message(WARNING "libtorch_python not found in ${PYTORCH_LIB_DIR}. This is often critical for PyTorch C++ extensions.")
        else()
            message(STATUS "Found libtorch_python: ${TORCH_PYTHON_LIBRARY_PATH}")
        endif()
        if(NOT (TORCH_LIBRARY_PATH OR TORCH_CPU_LIBRARY_PATH)) # Check if at least one main torch lib is found
            message(WARNING "Main PyTorch library (libtorch.so or libtorch_cpu.so) not found in ${PYTORCH_LIB_DIR}")
        else()
            if(TORCH_LIBRARY_PATH)
                message(STATUS "Found libtorch: ${TORCH_LIBRARY_PATH}")
            endif()
            if(TORCH_CPU_LIBRARY_PATH)
                message(STATUS "Found libtorch_cpu: ${TORCH_CPU_LIBRARY_PATH}")
            endif()
        endif()
        if(NOT C10_LIBRARY_PATH)
            message(WARNING "libc10 not found in ${PYTORCH_LIB_DIR}")
        else()
            message(STATUS "Found libc10: ${C10_LIBRARY_PATH}")
        endif()
    else()
        message(WARNING "Could not determine a valid PyTorch library directory (PYTORCH_LIB_DIR='${PYTORCH_LIB_DIR}'). Manual configuration might be needed.")
    endif()
else()
    message(WARNING "PyTorch not found (TORCH_FOUND is false) or PYTHON_EXECUTABLE not available. Cannot manually find PyTorch libraries.")
endif()

# --- Python Extension Module (gelib_base) ---

# 1. Identify Binding Source Files
set(GELIB_BINDING_SOURCES
    "${CMAKE_CURRENT_SOURCE_DIR}/python/bindings/GElib_py.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/include/GElib_base.cpp" # Compile GElib_base.cpp as part of the module
)

# Check if the main binding file exists
# Note: GELIB_BINDING_SOURCES is now a list, EXISTS checks the first element or if the var is defined.
# Better to check each file if necessary, or rely on build failure if a source is missing.
# For now, this check might be misleading or only check GElib_py.cpp.
if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/python/bindings/GElib_py.cpp")
    message(FATAL_ERROR "Main binding source file GElib_py.cpp not found.")
endif()

# 2. Define Python Extension Target
# We are not using pybind11_add_module here as setup.py used CppExtension directly,
# implying custom C++/ATen bindings.
add_library(gelib_base MODULE ${GELIB_BINDING_SOURCES})
set_target_properties(gelib_base PROPERTIES PREFIX "")

# Apply PyTorch includes and compile flags directly to gelib_base
if(TORCH_FOUND)
    if(TORCH_INCLUDE_DIRS)
        target_include_directories(gelib_base PRIVATE ${TORCH_INCLUDE_DIRS})
        message(STATUS "Added TORCH_INCLUDE_DIRS directly to gelib_base.")
    endif()
    if(TORCH_CXX_FLAGS)
        target_compile_options(gelib_base PRIVATE ${TORCH_CXX_FLAGS})
        message(STATUS "Added TORCH_CXX_FLAGS directly to gelib_base.")
    endif()
endif()

# Add Python include directories if Python found
if(Python3_FOUND AND Python3_INCLUDE_DIRS)
    target_include_directories(gelib_base SYSTEM PRIVATE ${Python3_INCLUDE_DIRS})
    target_compile_options(gelib_base PRIVATE "-I${Python3_INCLUDE_DIRS}") # Keep as forceful backup
    message(STATUS "Added Python3 include directories to gelib_base (SYSTEM): ${Python3_INCLUDE_DIRS}")
else()
    message(WARNING "Python3 development headers not found (Python3_INCLUDE_DIRS not set). Compilation of gelib_base may fail.")
endif()

# 3. Link against gelib_cpp_core (which brings cnine and PyTorch)
#    Also link directly against PyTorch libraries to ensure symbols for pybind11 casters are available.
# gelib_cpp_core should have PUBLIC/INTERFACE linkage for includes and libraries it uses.

# RPATH Logic for PyTorch
if(TORCH_FOUND)
    if(TARGET Torch::torch_python)
        get_target_property(TORCH_PYTHON_LIB_LOCATION Torch::torch_python IMPORTED_LOCATION_RELEASE)
        if(NOT TORCH_PYTHON_LIB_LOCATION)
            get_target_property(TORCH_PYTHON_LIB_LOCATION Torch::torch_python IMPORTED_LOCATION)
        endif()
        if(TORCH_PYTHON_LIB_LOCATION)
            get_filename_component(PYTORCH_LIB_DIR ${TORCH_PYTHON_LIB_LOCATION} DIRECTORY)
        endif()
    endif()

    if(NOT PYTORCH_LIB_DIR AND TARGET Torch::torch_cpu) # Fallback to torch_cpu if torch_python not found or location not resolved
        get_target_property(TORCH_CPU_LIB_LOCATION Torch::torch_cpu IMPORTED_LOCATION_RELEASE)
        if(NOT TORCH_CPU_LIB_LOCATION)
            get_target_property(TORCH_CPU_LIB_LOCATION Torch::torch_cpu IMPORTED_LOCATION)
        endif()
        if(TORCH_CPU_LIB_LOCATION)
            get_filename_component(PYTORCH_LIB_DIR ${TORCH_CPU_LIB_LOCATION} DIRECTORY)
        endif()
    endif()

    if(NOT PYTORCH_LIB_DIR AND TORCH_CMAKE_PREFIX_PATH AND EXISTS "${TORCH_CMAKE_PREFIX_PATH}/../../lib")
        set(PYTORCH_LIB_DIR "${TORCH_CMAKE_PREFIX_PATH}/../../lib") # Common structure for pip installed torch
        get_filename_component(PYTORCH_LIB_DIR ${PYTORCH_LIB_DIR} ABSOLUTE)
    elseif(NOT PYTORCH_LIB_DIR AND TORCH_CMAKE_PREFIX_PATH AND EXISTS "${TORCH_CMAKE_PREFIX_PATH}/lib")
        set(PYTORCH_LIB_DIR "${TORCH_CMAKE_PREFIX_PATH}/lib") # Common structure for conda installed torch
        get_filename_component(PYTORCH_LIB_DIR ${PYTORCH_LIB_DIR} ABSOLUTE)
    endif()

    if(PYTORCH_LIB_DIR AND EXISTS ${PYTORCH_LIB_DIR})
        message(STATUS "Determined PyTorch library directory for RPATH: ${PYTORCH_LIB_DIR}")
        set_property(TARGET gelib_base APPEND PROPERTY BUILD_RPATH "${PYTORCH_LIB_DIR}")
        set_property(TARGET gelib_base APPEND PROPERTY INSTALL_RPATH "${PYTORCH_LIB_DIR}") # For wheel
        # Also consider setting CMAKE_INSTALL_RPATH to this for broader effect if needed
    else()
        message(WARNING "Could not reliably determine PyTorch library directory for RPATH. PYTORCH_LIB_DIR='${PYTORCH_LIB_DIR}'. TORCH_CMAKE_PREFIX_PATH='${TORCH_CMAKE_PREFIX_PATH}'. Manual configuration might be needed if runtime errors persist.")
    endif()
else()
    message(WARNING "PyTorch not found, cannot set RPATH for PyTorch libraries.")
endif()

# Linking logic for gelib_base
set(GELIB_BASE_PRIVATE_LINK_LIBS pybind11::pybind11)
if(TARGET Torch::torch) # General, often includes python and c10 components
    list(APPEND GELIB_BASE_PRIVATE_LINK_LIBS Torch::torch)
    message(STATUS "Prepared to link gelib_base with Torch::torch (PRIVATE).")
else() # More specific components if Torch::torch isn't available or isn't comprehensive
    if(TARGET Torch::torch_cpu)
        list(APPEND GELIB_BASE_PRIVATE_LINK_LIBS Torch::torch_cpu)
        message(STATUS "Prepared to link gelib_base with Torch::torch_cpu (PRIVATE).")
    else()
        message(WARNING "Torch::torch_cpu target not found for gelib_base linkage.")
    endif()
    if(TARGET Torch::torch_python) # Essential for pybind casters
        list(APPEND GELIB_BASE_PRIVATE_LINK_LIBS Torch::torch_python)
        message(STATUS "Prepared to link gelib_base with Torch::torch_python (PRIVATE).")
    else()
        message(WARNING "Torch::torch_python target not found for gelib_base linkage. This might lead to runtime errors.")
    endif()
    if(TARGET Torch::c10) # Core C++ utilities
        list(APPEND GELIB_BASE_PRIVATE_LINK_LIBS Torch::c10)
        message(STATUS "Prepared to link gelib_base with Torch::c10 (PRIVATE).")
    else()
        message(WARNING "Torch::c10 target not found for gelib_base linkage.")
    endif()

    # Fallback if specific modern targets are missing but older TORCH_LIBRARIES variable exists
    if(NOT (TARGET Torch::torch_cpu OR TARGET Torch::torch_python OR TARGET Torch::c10) AND TORCH_LIBRARIES)
        list(APPEND GELIB_BASE_PRIVATE_LINK_LIBS ${TORCH_LIBRARIES})
        message(STATUS "Linking gelib_base with TORCH_LIBRARIES as fallback: ${TORCH_LIBRARIES} (PRIVATE).")
    endif()
endif()

target_link_libraries(gelib_base
    PUBLIC gelib_cpp_core # As per subtask instruction
    PRIVATE ${GELIB_BASE_PRIVATE_LINK_LIBS}
)
# Linking logic for gelib_base (revised for manual library paths)
set(GELIB_BASE_PRIVATE_LINK_LIBS_TEMP pybind11::pybind11) # Temporary list for PyTorch libs

if(TORCH_PYTHON_LIBRARY_PATH)
    list(APPEND GELIB_BASE_PRIVATE_LINK_LIBS_TEMP "${TORCH_PYTHON_LIBRARY_PATH}")
endif()
if(TORCH_CPU_LIBRARY_PATH) # Prefer specific torch_cpu if found
    list(APPEND GELIB_BASE_PRIVATE_LINK_LIBS_TEMP "${TORCH_CPU_LIBRARY_PATH}")
elseif(TORCH_LIBRARY_PATH) # Fallback to general torch lib if torch_cpu not found
    list(APPEND GELIB_BASE_PRIVATE_LINK_LIBS_TEMP "${TORCH_LIBRARY_PATH}")
endif()
if(C10_LIBRARY_PATH)
    list(APPEND GELIB_BASE_PRIVATE_LINK_LIBS_TEMP "${C10_LIBRARY_PATH}")
endif()

# Fallback to TORCH_LIBRARIES if manual finds yielded nothing substantial for core libs,
# or if TORCH_LIBRARIES contains other necessary system dependencies (like pthreads, etc.)
if(TORCH_LIBRARIES)
    # This ensures that components from TORCH_LIBRARIES that are not already found manually are added.
    # Useful if TORCH_LIBRARIES has more than just libtorch*, libc10*.
    list(APPEND GELIB_BASE_PRIVATE_LINK_LIBS_TEMP ${TORCH_LIBRARIES})
endif()
list(REMOVE_DUPLICATES GELIB_BASE_PRIVATE_LINK_LIBS_TEMP)

target_link_libraries(gelib_base
    PUBLIC gelib_cpp_core
    PRIVATE ${GELIB_BASE_PRIVATE_LINK_LIBS_TEMP}
)
message(STATUS "Linking gelib_base with PUBLIC gelib_cpp_core and PRIVATELY with: ${GELIB_BASE_PRIVATE_LINK_LIBS_TEMP}")

# RPATH setting using the PYTORCH_LIB_DIR determined by the execute_process block
if(PYTORCH_LIB_DIR AND EXISTS ${PYTORCH_LIB_DIR})
    message(STATUS "Setting BUILD_RPATH for gelib_base to include PyTorch directory: ${PYTORCH_LIB_DIR}")
    set_property(TARGET gelib_base APPEND PROPERTY BUILD_RPATH "${PYTORCH_LIB_DIR}")

    # For INSTALL_RPATH in the wheel, make it relative to the location of gelib_base.so
    # Assuming gelib_base.so is in .../site-packages/gelib/
    # and torch libs are in .../site-packages/torch/lib/
    # So, from gelib/, go up one level (to site-packages) then to torch/lib.
    # $ORIGIN refers to the directory of the executable/library at runtime.
    set_target_properties(gelib_base PROPERTIES
        INSTALL_RPATH "$ORIGIN/../torch/lib"
        INSTALL_RPATH_USE_LINK_PATH FALSE # We are manually setting INSTALL_RPATH
    )
    message(STATUS "Setting INSTALL_RPATH for gelib_base to $ORIGIN/../torch/lib")
else()
    message(WARNING "PYTORCH_LIB_DIR ('${PYTORCH_LIB_DIR}') is not valid or does not exist. RPATH for PyTorch libraries not optimally set for gelib_base wheel.")
endif()

# 4. Include Directories for bindings
# The necessary include directories (project-specific, cnine, PyTorch)
# should be inherited from gelib_cpp_core.
# Add python/bindings if it contains headers used by GElib_py.cpp itself.
target_include_directories(gelib_base PRIVATE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/python/bindings>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)

# 5. Ensure correct module naming for Python
# CMAKE_SHARED_MODULE_PREFIX and CMAKE_SHARED_MODULE_SUFFIX are usually
# correctly set by CMake for MODULE libraries for the current platform.
# Explicitly setting can be done if needed, but often not required.
# Example: set_target_properties(gelib_base PROPERTIES PREFIX "${PYTHON_MODULE_PREFIX}" SUFFIX "${PYTHON_MODULE_SUFFIX}")
# For scikit-build-core, ensuring it's a MODULE library is key.

message(STATUS "Defined Python extension module target: gelib_base")

# --- Installation Rules for Python Module ---
# This ensures that scikit-build-core can package the gelib_base module correctly.
# The pyproject.toml is configured for the module to be in a 'gelib' package.
# (e.g., wheel contents: gelib/gelib_base.so)

if(TARGET gelib_base)
    include(GNUInstallDirs) # Provides CMAKE_INSTALL_LIBDIR etc. but we use a relative path.

    # Install the Python module (e.g., gelib_base.so) into a 'gelib' directory
    # relative to the install prefix. This matches the package structure expected
    # by scikit-build-core based on pyproject.toml settings (packages = ["gelib"]).
    install(TARGETS gelib_base
            # LIBRARY DESTINATION will place it in <prefix>/<CMAKE_INSTALL_LIBDIR>/gelib
            # MODULE DESTINATION is often more appropriate for Python modules.
            # For scikit-build-core, a relative path like 'gelib' is common.
            LIBRARY DESTINATION "gelib"  # Creates <prefix>/gelib/gelib_base.so
            # MODULE DESTINATION "gelib"   # This line was causing an error
            RUNTIME DESTINATION "gelib"  # For Windows DLLs if any, good practice
           )
    message(STATUS "Install rules added for target: gelib_base into 'gelib' directory.")
else()
    message(WARNING "Target gelib_base not found. Cannot add install rules.")
endif()

message(STATUS "Configuring C++ tests...")
add_subdirectory(O3/tests)
add_subdirectory(SO3/tests)
