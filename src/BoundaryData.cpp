#include <Kokkos_Core.hpp>

#include <mpi.h>

#include "Dimension.H"
#include "Types.H"
#include "ProblemSetup.H"
#include "BoundaryData.H"

namespace KFVM {

  BoundaryData::BoundaryData(const ProblemSetup& ps):
    wCellSendLo({KFVM_D_DECL(ps.rad  ,ps.rad        ,ps.rad        )}),
    wCellSendUp({KFVM_D_DECL(2*ps.rad,ps.nY + ps.rad,ps.nZ + ps.rad)}),
    wCellRecvLo({KFVM_D_DECL(0       ,ps.rad        ,ps.rad        )}),
    wCellRecvUp({KFVM_D_DECL(ps.rad  ,ps.nY + ps.rad,ps.nZ + ps.rad)}),
    eCellSendLo({KFVM_D_DECL(ps.nX           ,ps.rad        ,ps.rad        )}),
    eCellSendUp({KFVM_D_DECL(ps.nX + ps.rad  ,ps.nY + ps.rad,ps.nZ + ps.rad)}),
    eCellRecvLo({KFVM_D_DECL(ps.nX + ps.rad  ,ps.rad        ,ps.rad        )}),
    eCellRecvUp({KFVM_D_DECL(ps.nX + 2*ps.rad,ps.nY + ps.rad,ps.nZ + ps.rad)}),
    sCellSendLo({KFVM_D_DECL(0               ,ps.rad  ,ps.rad        )}),
    sCellSendUp({KFVM_D_DECL(ps.nX + 2*ps.rad,2*ps.rad,ps.nZ + ps.rad)}),
    sCellRecvLo({KFVM_D_DECL(0               ,0       ,ps.rad        )}),
    sCellRecvUp({KFVM_D_DECL(ps.nX + 2*ps.rad,ps.rad  ,ps.nZ + ps.rad)}),
    nCellSendLo({KFVM_D_DECL(0               ,ps.nY           ,ps.rad        )}),
    nCellSendUp({KFVM_D_DECL(ps.nX + 2*ps.rad,ps.nY + ps.rad  ,ps.nZ + ps.rad)}),
    nCellRecvLo({KFVM_D_DECL(0               ,ps.nY + ps.rad  ,ps.rad        )}),
    nCellRecvUp({KFVM_D_DECL(ps.nX + 2*ps.rad,ps.nY + 2*ps.rad,ps.nZ + ps.rad)}),
    bCellSendLo({KFVM_D_DECL(0               ,0               ,ps.rad  )}),
    bCellSendUp({KFVM_D_DECL(ps.nX + 2*ps.rad,ps.nY + 2*ps.rad,2*ps.rad)}),
    bCellRecvLo({KFVM_D_DECL(0               ,0               ,0       )}),
    bCellRecvUp({KFVM_D_DECL(ps.nX + 2*ps.rad,ps.nY + 2*ps.rad,ps.rad  )}),
    tCellSendLo({KFVM_D_DECL(0               ,0               ,ps.nZ           )}),
    tCellSendUp({KFVM_D_DECL(ps.nX + 2*ps.rad,ps.nY + 2*ps.rad,ps.nZ + ps.rad  )}),
    tCellRecvLo({KFVM_D_DECL(0               ,0               ,ps.nZ + ps.rad  )}),
    tCellRecvUp({KFVM_D_DECL(ps.nX + 2*ps.rad,ps.nY + 2*ps.rad,ps.nZ + 2*ps.rad)})
  {
    // Allocate and assign views as needed
    if (ps.layoutMPI.wDst != MPI_PROC_NULL) {
      wCellSend = ConsDataPinnedView("Solver::BoundaryData::wCellSend",
				     KFVM_D_DECL(ps.rad,ps.nY,ps.nZ));
      wFaceSend = FaceDataPinnedView("Solver::BoundaryData::wFaceSend",
				     KFVM_DM_DECL(ps.nY,ps.nZ));
    }
    
    if (ps.layoutMPI.wSrc != MPI_PROC_NULL) {
      wCellRecv = ConsDataPinnedView("Solver::BoundaryData::wCellRecv",
				     KFVM_D_DECL(ps.rad,ps.nY,ps.nZ));
      wFaceRecv = FaceDataPinnedView("Solver::BoundaryData::wFaceRecv",
				     KFVM_DM_DECL(ps.nY,ps.nZ));
    }
    
    if (ps.layoutMPI.eDst != MPI_PROC_NULL) {
      eCellSend = ConsDataPinnedView("Solver::BoundaryData::eCellSend",
				     KFVM_D_DECL(ps.rad,ps.nY,ps.nZ));
      eFaceSend = FaceDataPinnedView("Solver::BoundaryData::eFaceSend",
				     KFVM_DM_DECL(ps.nY,ps.nZ));
    }
    
    if (ps.layoutMPI.eSrc != MPI_PROC_NULL) {
      eCellRecv = ConsDataPinnedView("Solver::BoundaryData::eCellRecv",
				     KFVM_D_DECL(ps.rad,ps.nY,ps.nZ));
      eFaceRecv = FaceDataPinnedView("Solver::BoundaryData::eFaceRecv",
				     KFVM_DM_DECL(ps.nY,ps.nZ));
    }
    
    if (ps.layoutMPI.sDst != MPI_PROC_NULL) {
      sCellSend = ConsDataPinnedView("Solver::BoundaryData::sCellSend",
				     KFVM_D_DECL(ps.nX + 2*ps.rad,ps.rad,ps.nZ));
      sFaceSend = FaceDataPinnedView("Solver::BoundaryData::sFaceSend",
				     KFVM_DM_DECL(ps.nX,ps.nZ));
    }
    
    if (ps.layoutMPI.sSrc != MPI_PROC_NULL) {
      sCellRecv = ConsDataPinnedView("Solver::BoundaryData::sCellRecv",
				     KFVM_D_DECL(ps.nX + 2*ps.rad,ps.rad,ps.nZ));
      sFaceRecv = FaceDataPinnedView("Solver::BoundaryData::sFaceRecv",
				     KFVM_DM_DECL(ps.nX,ps.nZ));
    }
    
    if (ps.layoutMPI.nDst != MPI_PROC_NULL) {
      nCellSend = ConsDataPinnedView("Solver::BoundaryData::nCellSend",
				     KFVM_D_DECL(ps.nX + 2*ps.rad,ps.rad,ps.nZ));
      nFaceSend = FaceDataPinnedView("Solver::BoundaryData::nFaceSend",
				     KFVM_DM_DECL(ps.nX,ps.nZ));
    }
    
    if (ps.layoutMPI.nSrc != MPI_PROC_NULL) {
      nCellRecv = ConsDataPinnedView("Solver::BoundaryData::nCellRecv",
				     KFVM_D_DECL(ps.nX + 2*ps.rad,ps.rad,ps.nZ));
      nFaceRecv = FaceDataPinnedView("Solver::BoundaryData::nFaceRecv",
				     KFVM_DM_DECL(ps.nX,ps.nZ));
    }
    
    if (ps.layoutMPI.bDst != MPI_PROC_NULL) {
      bCellSend = ConsDataPinnedView("Solver::BoundaryData::bCellSend",
				     KFVM_D_DECL(ps.nX + 2*ps.rad,ps.nY + 2*ps.rad,ps.rad));
      bFaceSend = FaceDataPinnedView("Solver::BoundaryData::bFaceSend",
				     KFVM_DM_DECL(ps.nX,ps.nY));
    }
    
    if (ps.layoutMPI.bSrc != MPI_PROC_NULL) {
      bCellRecv = ConsDataPinnedView("Solver::BoundaryData::bCellRecv",
				     KFVM_D_DECL(ps.nX + 2*ps.rad,ps.nY + 2*ps.rad,ps.rad));
      bFaceRecv = FaceDataPinnedView("Solver::BoundaryData::bFaceRecv",
				     KFVM_DM_DECL(ps.nX,ps.nY));
    }
    
    if (ps.layoutMPI.tDst != MPI_PROC_NULL) {
      tCellSend = ConsDataPinnedView("Solver::BoundaryData::tCellSend",
				     KFVM_D_DECL(ps.nX + 2*ps.rad,ps.nY + 2*ps.rad,ps.rad));
      tFaceSend = FaceDataPinnedView("Solver::BoundaryData::tFaceSend",
				     KFVM_DM_DECL(ps.nX,ps.nY));
    }
    
    if (ps.layoutMPI.tSrc != MPI_PROC_NULL) {
      tCellRecv = ConsDataPinnedView("Solver::BoundaryData::tCellRecv",
				     KFVM_D_DECL(ps.nX + 2*ps.rad,ps.nY + 2*ps.rad,ps.rad));
      tFaceRecv = FaceDataPinnedView("Solver::BoundaryData::tFaceRecv",
				     KFVM_DM_DECL(ps.nX,ps.nY));
    }
  }
  
} // end namespace KFVM
