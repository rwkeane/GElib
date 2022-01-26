// This file is part of GElib, a C++/CUDA library for group
// equivariant tensor operations. 
// 
// Copyright (c) 2022, Imre Risi Kondor and Erik H Thiede
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.


#ifndef _SO3Fpart
#define _SO3Fpart

#include "CtensorB.hpp"
#include "SO3partB.hpp"
#include "SO3Fpart3_view.hpp"
#include "SO3Fpart_addFproductFn.hpp"
#include "SO3Fpart_addFproduct_back0Fn.hpp"
#include "SO3Fpart_addFproduct_back1Fn.hpp"
//#include "SO3_CGbank.hpp"
//#include "SO3_SPHgen.hpp"
//#include "SO3element.hpp"
//#include "WignerMatrix.hpp"

extern GElib::SO3_CGbank SO3_cgbank;
extern GElib::SO3_SPHgen SO3_sphGen;



namespace GElib{
  

  // An SO3Fpart is a  b x (2l+1) x (2l+1)  dimensional complex tensor.


  class SO3Fpart: public SO3partB{
  public:

    typedef cnine::device device;
    typedef cnine::fill_pattern fill_pattern;

    //using CtensorB::CtensorB;


  public: // ---- Constructors -------------------------------------------------------------------------------
    

    SO3Fpart(const int b, const int l, const int _dev=0):
      SO3partB(b,l,2*l+1,_dev){}
    
    template<typename FILLTYPE, typename = typename std::enable_if<std::is_base_of<fill_pattern, FILLTYPE>::value, FILLTYPE>::type>
    SO3Fpart(const int b, const int l, const FILLTYPE& dummy, const int _dev=0):
      SO3partB(b,l,2*l+1,dummy,_dev){}

    
  public: // ---- Named constructors -------------------------------------------------------------------------


    static SO3Fpart zero(const int b, const int l, const int _dev=0){
      return SO3Fpart(b,l,cnine::fill_zero(),_dev);
    }

    static SO3Fpart gaussian(const int b, const int l, const int _dev=0){
      return SO3Fpart(b,l,cnine::fill_gaussian(),_dev);
    }


  public: // ---- Access -------------------------------------------------------------------------------------

    
    //int getl() const{
    //return (dims(1)-1)/2;
    //}

    

  public: // ---- CG-products --------------------------------------------------------------------------------


    void add_FourierSpaceProduct(const SO3Fpart& x, const SO3Fpart& y){

      /*
      SO3part buf=SO3Fpart::CGproduct(x,y,getl());

      const int l=getl(); 
      const int l1=x.getl(); 
      const int l2=y.getl(); 
      const int N1=2*l1+1;
      const int N2=2*l2+1;
      const int nblocks=getn()/(2*getl()+1);
      auto& C=SO3_cgbank.getf(CGindex(l1,l2,l));

      int offs=0;
      SO3partA buf(l,N1*N2,cnine::fill::zero);
      for(int bl=0; bl<nblocks; bl++){
	for(int n1=0; n1<N1; n1++){
	  for(int n2=0; n2<N2; n2++){
	    for(int m1=-l1; m1<=l1; m1++){
	      for(int m2=std::max(-l2,-l-m1); m2<=std::min(l2,l-m1); m2++){
		buf.inc(offs+n2,m1+m2+l,C(m1+l1,m2+l2)*x(n1+bl*N1,m1+l1)*y(n2+bl*N2,m2+l2));
	      }
	    }
	  }
	  offs+=N2;
	}
      }

      int mult=2*l2+1;
      float fact=(float)((2*l1+1)*(2*l2+1))/(2*l+1);
      for(int bl=0; bl<nblocks; bl++){
	for(int n=0; n<2*l+1; n++){
	  for(int m1=-l1; m1<=l1; m1++){
	    for(int m2=std::max(-l2,-l-m1); m2<=std::min(l2,l-m1); m2++){
	      inc(m1+m2+l,n,C(m1+l1,m2+l2)*buf((m1+l1)*mult+m2+l2,n)*fact);
	    }
	  }
	}
      }
      */

    }


    void add_Fproduct(const SO3Fpart& x, const SO3Fpart& y){
      auto v=this->view();
      SO3Fpart_addFproductFn()(v,x,y);
    }

    void add_Fproduct_back0(const SO3Fpart& g, const SO3Fpart& y){
      auto v=this->view();
      SO3Fpart_addFproduct_back0Fn()(v,g,y);
    }

    void add_Fproduct_back1(const SO3Fpart& g, const SO3Fpart& x){
      auto v=this->view();
      SO3part_addFproduct_back0Fn()(v,g,x);
    }


  };


  const SO3Fpart& as_SO3Fpart(const SO3partB& x){
    return static_cast<const SO3Fpart&>(x);
  }

  SO3Fpart& as_SO3Fpart(SO3partB& x){
    return static_cast<SO3Fpart&>(x);
  }

}

#endif
