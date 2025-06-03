cmake_minimum_required(VERSION 3.18)
project(gelib LANGUAGES CXX)
enable_testing()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON) # Useful for debugging and IDEs

# Placeholder for future CUDA integration
option(GELIB_WITH_CUDA "Enable CUDA support (manual integration needed)" OFF)

# --- Add further CMake configurations below ---
# Add cnine submodule
add_subdirectory(deps/cnine)

# --- cnine Override Mechanism ---
# This section must be AFTER add_subdirectory(deps/cnine)
# It assumes 'cnine' is the target name defined by deps/cnine/CMakeLists.txt
if(TARGET cnine)
    message(STATUS "Configuring cnine overrides for GElib...")

    # 1. Override Include Directories
    # Prepend paths from cnine_overrides_for_gelib to cnine's include directories.
    # We assume that headers in cnine_overrides_for_gelib follow the same structure
    # as in deps/cnine/include or deps/cnine/src.
    # Example: if overriding deps/cnine/src/foo/bar.hpp, it would be
    # cnine_overrides_for_gelib/src/foo/bar.hpp
    target_include_directories(cnine BEFORE INTERFACE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/cnine_overrides_for_gelib>
        # Add specific subdirectories if cnine's includes are more structured, e.g.:
        # $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/cnine_overrides_for_gelib/include>
        # $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/cnine_overrides_for_gelib/src>
    )
    message(STATUS "cnine target_include_directories prepended with cnine_overrides_for_gelib.")

    # 2. Override Source Files
    # Get the original list of sources for the 'cnine' target
    get_target_property(CNINE_ORIGINAL_SOURCES cnine SOURCES)
    # message(STATUS "Original cnine sources: ${CNINE_ORIGINAL_SOURCES}") # For debugging

    set(CNINE_MODIFIED_SOURCES "")
    set(CNINE_OVERRIDE_BASE_PATH_ABS ${CMAKE_CURRENT_SOURCE_DIR}/cnine_overrides_for_gelib)
    set(CNINE_DEPS_BASE_PATH_ABS ${CMAKE_CURRENT_SOURCE_DIR}/deps/cnine)

    foreach(SOURCE_FILE ${CNINE_ORIGINAL_SOURCES})
        # Make the source file path absolute if it's not already
        if(IS_ABSOLUTE ${SOURCE_FILE})
            set(ABS_SOURCE_FILE ${SOURCE_FILE})
        else()
            set(ABS_SOURCE_FILE ${CMAKE_CURRENT_BINARY_DIR}/${SOURCE_FILE}) # Assuming sources are relative to cnine's build dir
            # A more robust way for sources from add_subdirectory is if they are relative to that subdirectory's source.
            # However, get_target_property(SOURCES) often returns paths relative to where cmake is run or the target's build dir.
            # For sources from deps/cnine, they are likely relative to deps/cnine source dir or build dir.
            # Let's try to reconstruct potential original path relative to cnine's source dir
            # This is tricky because CMake might make paths relative to current binary dir
            # For now, we focus on replacing based on the basename and expected subdirectory structure.

            # Heuristic: Check if path starts with cnine's source or build dir path within deps.
            # This part is complex as source paths can be relative.
            # A simpler approach: get the part of the path relative to 'deps/cnine/'
            string(REPLACE "${CNINE_DEPS_BASE_PATH_ABS}/" "" RELATIVE_SOURCE_FILE ${ABS_SOURCE_FILE})
            # This might not always work if ABS_SOURCE_FILE is not under CNINE_DEPS_BASE_PATH_ABS
            # For files coming from target_sources, they might be already processed.
            # A common case: sources are listed relative to deps/cnine/CMakeLists.txt location
            # So, if a source is "src/file.cpp", the full path is "deps/cnine/src/file.cpp"
            # Let's assume SOURCE_FILE is relative to deps/cnine source directory
            set(POTENTIAL_ORIGINAL_PATH_IN_DEPS ${CNINE_DEPS_BASE_PATH_ABS}/${SOURCE_FILE})
            string(REPLACE "${CNINE_DEPS_BASE_PATH_ABS}/" "" RELATIVE_TO_CNINE_ROOT ${POTENTIAL_ORIGINAL_PATH_IN_DEPS})
        endif()

        # If SOURCE_FILE is already absolute, try to make it relative to deps/cnine root
        if(IS_ABSOLUTE ${SOURCE_FILE})
            if(${SOURCE_FILE} MATCHES "^${CNINE_DEPS_BASE_PATH_ABS}")
                 string(REPLACE "${CNINE_DEPS_BASE_PATH_ABS}/" "" RELATIVE_TO_CNINE_ROOT ${SOURCE_FILE})
            else()
                # If absolute but not in deps/cnine, it's probably a generated file or from elsewhere. Don't try to override.
                set(RELATIVE_TO_CNINE_ROOT "") # Mark as not overridable this way
            endif()
        else() # If relative, assume it's relative to cnine's source directory (CMAKE_CURRENT_SOURCE_DIR within add_subdirectory)
            set(RELATIVE_TO_CNINE_ROOT ${SOURCE_FILE})
        endif()

        set(OVERRIDE_FILE_PATH "")
        if(NOT CMAKE_MATCH_1 STREQUAL "" AND RELATIVE_TO_CNINE_ROOT) # Ensure RELATIVE_TO_CNINE_ROOT is not empty
            set(OVERRIDE_FILE_PATH ${CNINE_OVERRIDE_BASE_PATH_ABS}/${RELATIVE_TO_CNINE_ROOT})
        endif()

        if(EXISTS ${OVERRIDE_FILE_PATH})
            list(APPEND CNINE_MODIFIED_SOURCES ${OVERRIDE_FILE_PATH})
            message(STATUS "  Overriding cnine source: ${RELATIVE_TO_CNINE_ROOT} with ${OVERRIDE_FILE_PATH}")
        else()
            list(APPEND CNINE_MODIFIED_SOURCES ${SOURCE_FILE}) # Keep original
        endif()
    endforeach()

    if(CNINE_MODIFIED_SOURCES)
        # Clear existing sources and add the new list.
        # Using target_sources() with PRIVATE/PUBLIC/INTERFACE can be complex for replacing.
        # Setting the SOURCES property directly is more straightforward for complete replacement.
        # However, target_sources() is the modern way to manipulate sources.
        # Let's try clearing and then adding.
        # First, ensure the property is settable (it should be for SOURCES)
        # set_target_properties(cnine PROPERTIES SOURCES "") # This might not work or be disallowed.

        # Alternative: If cnine uses a variable like CNINE_SOURCES internally, we can't easily modify it post-add_subdirectory.
        # The most reliable method if just replacing the list is needed:
        # This is a common approach but check CMake version compatibility for clearing sources this way.
        # For CMake 3.13+, target_sources can clear:
        # target_sources(cnine PRIVATE "") # This would remove all sources if it worked as expected for clearing.
        # Then add them back:
        # target_sources(cnine PRIVATE ${CNINE_MODIFIED_SOURCES})
        # However, the documentation for target_sources mentions it's for adding/removing specific files.
        #
        # A safer bet for wholesale replacement might be if cnine uses an object library,
        # or if we have to resort to creating our own target for cnine.
        #
        # Given the constraints, directly setting the SOURCES property is often used,
        # though it's a bit of a blunt instrument.
        # Let's assume for now cnine is a normal library (STATIC/SHARED) and try this.
        # If this doesn't work, we'd need to investigate cnine's CMake more deeply or use Option C.

        # The most common way to set sources for a target if it's already defined
        # is to ensure they are correctly passed when the target is created.
        # Modifying them afterwards with set_target_properties for SOURCES is possible.

        # Let's try to use target_sources with a generator expression to filter. This is advanced.
        # A simpler approach for now: if the list of sources is different, update the property.
        if(NOT "${CNINE_MODIFIED_SOURCES}" STREQUAL "${CNINE_ORIGINAL_SOURCES}")
            # This is the part that's hard to make robust without knowing cnine's internals.
            # If 'cnine' is an ALIAS or INTERFACE library, this will fail. Assuming it's STATIC/SHARED.
            # The most direct way to replace sources:
            # First, clear the SOURCES property. This is not directly possible.
            # Instead, we add the new sources. If the old sources were added with target_sources(),
            # they might persist.
            # The behavior of add_library() vs target_sources() is key here.
            # If add_library(cnine ${SOME_SOURCES_VAR}) was used, then CNINE_ORIGINAL_SOURCES will contain those.
            #
            # For now, the most practical approach if `cnine` is a simple library target:
            # `add_library(cnine_placeholder TYPE UNKNOWN IMPORTED)` # Create a dummy target
            # `set_property(TARGET cnine PROPERTY SOURCES ${CNINE_MODIFIED_SOURCES})` # This should work
            # This will effectively replace the source list for compilation.
            #
            # A better way could be to create a new target that wraps/replaces cnine if direct modification is too hard.
            #
            # Let's assume `set_target_properties` can update the list.
            # This is the most common documented way to change properties.
            message(STATUS "Attempting to set modified sources for cnine target.")
            set_target_properties(cnine PROPERTIES SOURCES "${CNINE_MODIFIED_SOURCES}")
            # Verify if it changed (for debugging, not in final script)
            # get_target_property(CNINE_POST_SOURCES cnine SOURCES)
            # message(STATUS "Sources for cnine after modification attempt: ${CNINE_POST_SOURCES}")
        else()
            message(STATUS "No cnine source file overrides found or sources remain unchanged.")
        endif()
    else()
        message(STATUS "Original cnine sources list was empty or not retrieved. Skipping source override.")
    endif()
else()
    message(WARNING "cnine target not found. Cannot apply overrides.")
endif()
# --- End cnine Override Mechanism ---


# --- Core C++ Library (gelib_cpp_core) ---

# This library is an INTERFACE library. It does not have its own sources to compile.
# It's used to group include directories, compile definitions, and link dependencies.
# Removing the source file globbing and warning for GELIB_CORE_SOURCES as it's expected to be empty.
# file(GLOB_RECURSE GELIB_CORE_SOURCES_ALL
#     "${CMAKE_CURRENT_SOURCE_DIR}/core/*.cpp"
#     "${CMAKE_CURRENT_SOURCE_DIR}/SO3/*.cpp"
#     "${CMAKE_CURRENT_SOURCE_DIR}/O3/*.cpp"
# )
# set(GELIB_CORE_SOURCES "")
# foreach(FILE ${GELIB_CORE_SOURCES_ALL})
#     if(NOT FILE MATCHES "\\.cu(\\.cc)?$") # Escaped backslashes for literal dot
#         list(APPEND GELIB_CORE_SOURCES ${FILE})
#     endif()
# endforeach()
# if(NOT GELIB_CORE_SOURCES)
#     message(STATUS "gelib_cpp_core is an INTERFACE library with no dedicated source files, as expected.")
# else()
#     # This case should ideally not happen if it's a pure interface library.
#     message(WARNING "gelib_cpp_core has sources listed, but is defined as INTERFACE. Check definition if these need compilation for gelib_cpp_core itself: ${GELIB_CORE_SOURCES}")
# endif()

add_library(gelib_cpp_core INTERFACE)
target_compile_features(gelib_cpp_core INTERFACE cxx_std_17)

target_include_directories(gelib_cpp_core INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/core>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/SO3>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/O3>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/cuda>
)

if(TARGET cnine)
    target_link_libraries(gelib_cpp_core INTERFACE cnine)
else()
    message(WARNING "cnine target not found. Include directories from cnine might be missing for gelib_cpp_core.")
endif()

# 5. Compiler Flags and Definitions
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
target_compile_options(gelib_cpp_core INTERFACE ${GELIB_CXX_COMPILE_FLAGS})

set(GELIB_CXX_DEFINITIONS
    _WITH_ATEN # PyTorch related
    CNINE_RANGE_CHECKING
    CNINE_SIZE_CHECKING
    CNINE_DEVICE_CHECKING
    GELIB_RANGE_CHECKING
    WITH_FAKE_GRAD
    GELIB_COPY_WARNINGS
    GELIB_MOVE_WARNINGS
    GELIB_CONVERT_WARNINGS
    CNINE_ATEN_CONVERT_WARNINGS
)
target_compile_definitions(gelib_cpp_core INTERFACE ${GELIB_CXX_DEFINITIONS})

if(GELIB_WITH_CUDA)
    target_compile_definitions(gelib_cpp_core INTERFACE _WITH_CUDA _WITH_CUBLAS)
endif()

# --- PyTorch Integration ---
# This relies on CMAKE_PREFIX_PATH being set correctly to find PyTorch
# in the current Python environment. scikit-build-core should handle this,
# or it can be set manually:
# e.g., cmake -DCMAKE_PREFIX_PATH=$(python -c 'import torch; print(torch.utils.cmake_prefix_path)') ...
find_package(Torch REQUIRED)

# --- Python Interpreter and Development Headers ---
# Find Python development headers, which are required for pybind11
find_package(Python COMPONENTS Interpreter Development REQUIRED)

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

# --- Python Extension Module (gelib_base) ---

# 1. Identify Binding Source Files
# Based on setup.py, GElib_py.cpp is the main entry point for the 'gelib_base' extension.
# It's assumed that if other _py.cpp files in python/bindings/ are needed by gelib_base,
# they are included by GElib_py.cpp or compiled separately if they were part of other extensions.
# Only GElib_py.cpp should be listed as a source file for the target.
set(GELIB_BINDING_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/python/bindings/GElib_py.cpp")

# Check if the main binding file exists
if(NOT EXISTS ${GELIB_BINDING_SOURCES}) # Check existence of the specific file
    message(FATAL_ERROR "Main binding source file GElib_py.cpp not found in python/bindings/")
endif()

# 2. Define Python Extension Target
# We are not using pybind11_add_module here as setup.py used CppExtension directly,
# implying custom C++/ATen bindings.
add_library(gelib_base MODULE ${GELIB_BINDING_SOURCES})

# 3. Link against gelib_cpp_core (which brings cnine and PyTorch)
# gelib_cpp_core should have PUBLIC/INTERFACE linkage for includes and libraries it uses.
target_link_libraries(gelib_base PRIVATE gelib_cpp_core)

# 4. Include Directories for bindings
# The necessary include directories (project-specific, cnine, PyTorch)
# should be inherited from gelib_cpp_core.
# Add python/bindings if it contains headers used by GElib_py.cpp itself.
target_include_directories(gelib_base PRIVATE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/python/bindings>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${Python_INCLUDE_DIRS}> # Add Python include directory
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
