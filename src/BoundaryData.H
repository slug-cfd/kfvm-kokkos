#ifndef BOUNDARYDATA_H_
#define BOUNDARYDATA_H_

#include <Kokkos_Core.hpp>

#include <Definitions.H>

#include "Dimension.H"
#include "ProblemSetup.H"
#include "Types.H"

namespace KFVM {

struct BoundaryData {
  // Shape of each buffer
  const Kokkos::Array<idx_t, SPACE_DIM> wCellSendLo, wCellSendUp;
  const Kokkos::Array<idx_t, SPACE_DIM> wCellRecvLo, wCellRecvUp;
  const Kokkos::Array<idx_t, SPACE_DIM> eCellSendLo, eCellSendUp;
  const Kokkos::Array<idx_t, SPACE_DIM> eCellRecvLo, eCellRecvUp;
  const Kokkos::Array<idx_t, SPACE_DIM> sCellSendLo, sCellSendUp;
  const Kokkos::Array<idx_t, SPACE_DIM> sCellRecvLo, sCellRecvUp;
  const Kokkos::Array<idx_t, SPACE_DIM> nCellSendLo, nCellSendUp;
  const Kokkos::Array<idx_t, SPACE_DIM> nCellRecvLo, nCellRecvUp;
  const Kokkos::Array<idx_t, SPACE_DIM> bCellSendLo, bCellSendUp;
  const Kokkos::Array<idx_t, SPACE_DIM> bCellRecvLo, bCellRecvUp;
  const Kokkos::Array<idx_t, SPACE_DIM> tCellSendLo, tCellSendUp;
  const Kokkos::Array<idx_t, SPACE_DIM> tCellRecvLo, tCellRecvUp;

  // Cell data buffers, only allocated if needed
  ConsDataPinnedView wCellSend, wCellRecv, eCellSend, eCellRecv;
  ConsDataPinnedView sCellSend, sCellRecv, nCellSend, nCellRecv;
  ConsDataPinnedView bCellSend, bCellRecv, tCellSend, tCellRecv;

  // Face data buffers, only allocated if needed
  FaceDataPinnedView wFaceSend, wFaceRecv, eFaceSend, eFaceRecv;
  FaceDataPinnedView sFaceSend, sFaceRecv, nFaceSend, nFaceRecv;
  FaceDataPinnedView bFaceSend, bFaceRecv, tFaceSend, tFaceRecv;

  BoundaryData(const ProblemSetup &ps_);
};

} // end namespace KFVM
#endif
