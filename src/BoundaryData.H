#ifndef BOUNDARYDATA_H_
#define BOUNDARYDATA_H_

#include <Kokkos_Core.hpp>

#include <Definitions.H>

#include "Dimension.H"
#include "Types.H"
#include "ProblemSetup.H"

namespace KFVM {
  
  struct BoundaryData {
    // Shape of each buffer
    const Kokkos::Array<idx_t,SPACE_DIM> wCellSrcLo,wCellSrcUp;
    const Kokkos::Array<idx_t,SPACE_DIM> wCellDstLo,wCellDstUp;
    const Kokkos::Array<idx_t,SPACE_DIM> eCellSrcLo,eCellSrcUp;
    const Kokkos::Array<idx_t,SPACE_DIM> eCellDstLo,eCellDstUp;
    const Kokkos::Array<idx_t,SPACE_DIM> sCellSrcLo,sCellSrcUp;
    const Kokkos::Array<idx_t,SPACE_DIM> sCellDstLo,sCellDstUp;
    const Kokkos::Array<idx_t,SPACE_DIM> nCellSrcLo,nCellSrcUp;
    const Kokkos::Array<idx_t,SPACE_DIM> nCellDstLo,nCellDstUp;
    const Kokkos::Array<idx_t,SPACE_DIM> bCellSrcLo,bCellSrcUp;
    const Kokkos::Array<idx_t,SPACE_DIM> bCellDstLo,bCellDstUp;
    const Kokkos::Array<idx_t,SPACE_DIM> tCellSrcLo,tCellSrcUp;
    const Kokkos::Array<idx_t,SPACE_DIM> tCellDstLo,tCellDstUp;

    // Cell data buffers, only allocated if needed
    ConsDataCommView wCellSrc,wCellDst,eCellSrc,eCellDst;
    ConsDataCommView sCellSrc,sCellDst,nCellSrc,nCellDst;
    ConsDataCommView bCellSrc,bCellDst,tCellSrc,tCellDst;

    // Face data buffers, only allocated if needed
    FaceDataCommView wFaceSrc,wFaceDst,eFaceSrc,eFaceDst;
    FaceDataCommView sFaceSrc,sFaceDst,nFaceSrc,nFaceDst;
    FaceDataCommView bFaceSrc,bFaceDst,tFaceSrc,tFaceDst;
    
    BoundaryData(const ProblemSetup& ps_);
  };

} // end namespace KFVM
#endif
