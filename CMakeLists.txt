# To build:
#mkdir build
#cd build
#cmake ..
#cmake --build . --config Release

cmake_minimum_required(VERSION 3.15)
project(llama2 C)

#option(USE_PCRE "Enable PCRE regex support" ON)
option(USE_PCRE "Enable PCRE regex support" OFF)

# Find and enable OpenMP
find_package(OpenMP REQUIRED)

# Add the executable
add_executable(llama2 run.c win.c)

# Link OpenMP
target_link_libraries(llama2 PRIVATE OpenMP::OpenMP_C)

# Include current directory for headers
target_include_directories(llama2 PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

# PCRE configuration
if(USE_PCRE)
    message(STATUS "PCRE support enabled")
    
    # Define USE_PCRE=1 for the code
    target_compile_definitions(llama2 PRIVATE USE_PCRE=1)
    
    # Path to PCRE library and DLL
    set(PCRE_LIB_PATH "C:/Users/jonas/Downloads/pcre-8.45/pcre-8.45/build/Release/pcre.lib")
    set(PCRE_DLL_PATH "C:/Users/jonas/Downloads/pcre-8.45/pcre-8.45/build/Release/pcre.dll")
    
    # Link PCRE library
    target_link_libraries(llama2 PRIVATE ${PCRE_LIB_PATH})
    
    # Copy PCRE DLL to executable directory after build
    add_custom_command(TARGET llama2 POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            ${PCRE_DLL_PATH}
            $<TARGET_FILE_DIR:llama2>
        COMMENT "Copying pcre.dll to output directory"
    )
else()
    message(STATUS "PCRE support disabled (use -DUSE_PCRE=ON to enable)")
    
    # Define USE_PCRE=0 for the code
    target_compile_definitions(llama2 PRIVATE USE_PCRE=0)
endif()

# Set optimization flags based on compiler
if(MSVC)
    # Fast floating point and maximum optimization (equivalent to /fp:fast /Ox)
    target_compile_options(llama2 PRIVATE 
        $<$<CONFIG:Release>:/fp:fast /Ox>
        $<$<CONFIG:Debug>:/Od> # Disable optimization for Debug
    )
    
    # Alternative optimization levels:
    #target_compile_options(llama2 PRIVATE /fp:fast /O2) # Standard optimization
    #target_compile_options(llama2 PRIVATE /fp:fast /O1) # Minimize size
else()
    # GCC/Clang: Fast optimization (equivalent to /fp:fast /Ox)
    target_compile_options(llama2 PRIVATE 
        $<$<CONFIG:Release>:-Ofast>
        $<$<CONFIG:Debug>:-O0 -g> # No optimization, but with debug symbols
    )
    
    # Alternative optimization levels:
    #target_compile_options(llama2 PRIVATE -O3) # Aggressive optimization
    #target_compile_options(llama2 PRIVATE -O2) # Standard optimization
    #target_compile_options(llama2 PRIVATE -O1) # Basic optimization
endif()
