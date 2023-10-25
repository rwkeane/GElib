/*
 * This file is part of GElib, a C++/CUDA library for group equivariant 
 * tensor operations. 
 *  
 * Copyright (c) 2023, Imre Risi Kondor
 *
 * This source code file is subject to the terms of the noncommercial 
 * license distributed with GElib in the file NONCOMMERICAL.TXT. Commercial 
 * use is prohibited. All redistributed versions of this file (in orginal
 * or modified form) must retain this copyright notice and must be 
 * accompanied by a verbatim copy of the license. 
 *
 */

#include <cuda.h>
#include <cuda_runtime.h>
#include "GElib_base.hpp"

__device__ __constant__ unsigned char cg_cmem[CNINE_CONST_MEM_SIZE];
#define _SO3CG_CUDA_CONCAT

//#include "SO3partA_CGproduct.cu"
//#include "SO3partA_DiagCGproduct.cu"

#include "SO3partB_addCGproduct.cu"
#include "SO3partB_addCGproduct_back0.cu"
#include "SO3partB_addCGproduct_back1.cu"

#include "SO3partB_addDiagCGproduct.cu"
#include "SO3partB_addDiagCGproduct_back0.cu"
#include "SO3partB_addDiagCGproduct_back1.cu"

#include "SO3Fpart_addFproduct.cu"
#include "SO3Fpart_addFproduct_back0.cu"
#include "SO3Fpart_addFproduct_back1.cu"

#include "SO3part_addCGtransform.cu"

