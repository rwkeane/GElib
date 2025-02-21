// This file is part of GElib, a C++/CUDA library for group
// equivariant tensor operations. 
// 
// Copyright (c) 2024, Imre Risi Kondor
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.


#ifndef _SO3part_addDiagCGproduct_back1Fn
#define _SO3part_addDiagCGproduct_back1Fn

#include "GElib_base.hpp"
#include "SO3part.hpp"
#include "SO3CGbank.hpp"
#include "MultiLoop.hpp"
#include "WorkStreamLoop.hpp"

extern GElib::SO3CGbank SO3_CGbank;


namespace GElib{

#ifdef _WITH_CUDA
  void SO3part_addDiagCGproduct_back1_cu(SO3part y, SO3part r, SO3part x, const int offs, const cudaStream_t& stream);
#endif


  template<typename PART, typename TYPE>
  class SO3part_addDiagCGproduct_back1Fn{
  public:

    typedef cnine::TensorView<complex<TYPE> > TENSOR;

    void operator()(const PART& y, const PART& r, const PART& x, const int offs=0){
      const int l=r.getl(); 
      const int l1=x.getl(); 
      const int l2=y.getl();
 
      const int N=x.getn();
      GELIB_ASSRT(y.getn()==N);

      const int dev=r.dev;
      GELIB_ASSRT(x.get_dev()==dev);
      GELIB_ASSRT(x.get_dev()==dev);

      if(dev==0){
	auto& C=SO3_CGbank.get<TYPE>(l1,l2,l);
	y.for_each_cell_multi(r,x,[&](const int b, const int g, const TENSOR& y, const TENSOR& r, const TENSOR& x){
	    for(int n=0; n<N; n++){
	      for(int m1=-l1; m1<=l1; m1++){
		for(int m2=std::max(-l2,-l-m1); m2<=std::min(l2,l-m1); m2++){
		  y.inc(m2+l2,n,C(m1+l1,m2+l2)*r(m1+m2+l,offs+n)*std::conj(x(m1+l1,n)));
		}
	      }
	    }
	  });
      }

      if(dev==1){
	CUDA_STREAM(SO3part_addDiagCGproduct_cu(y,r,x,_offs,stream));
      }

    }


  };

}


#endif 

