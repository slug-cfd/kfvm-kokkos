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
      wCellSend = ConsDataCommView("Solver::BoundaryData::wCellSend",
				  KFVM_D_DECL(ps.rad,ps.nY,ps.nZ));
      wFaceSend = FaceDataCommView("Solver::BoundaryData::wFaceSend",
				  KFVM_DM_DECL(ps.nY,ps.nZ));
    }
    
    if (ps.layoutMPI.wSrc != MPI_PROC_NULL) {
      wCellRecv = ConsDataCommView("Solver::BoundaryData::wCellRecv",
				  KFVM_D_DECL(ps.rad,ps.nY,ps.nZ));
      wFaceRecv = FaceDataCommView("Solver::BoundaryData::wFaceRecv",
				  KFVM_DM_DECL(ps.nY,ps.nZ));
    }
    
    if (ps.layoutMPI.eDst != MPI_PROC_NULL) {
      eCellSend = ConsDataCommView("Solver::BoundaryData::eCellSend",
				  KFVM_D_DECL(ps.rad,ps.nY,ps.nZ));
      eFaceSend = FaceDataCommView("Solver::BoundaryData::eFaceSend",
				  KFVM_DM_DECL(ps.nY,ps.nZ));
    }
    
    if (ps.layoutMPI.eSrc != MPI_PROC_NULL) {
      eCellRecv = ConsDataCommView("Solver::BoundaryData::eCellRecv",
				  KFVM_D_DECL(ps.rad,ps.nY,ps.nZ));
      eFaceRecv = FaceDataCommView("Solver::BoundaryData::eFaceRecv",
				  KFVM_DM_DECL(ps.nY,ps.nZ));
    }
    
    if (ps.layoutMPI.sDst != MPI_PROC_NULL) {
      sCellSend = ConsDataCommView("Solver::BoundaryData::sCellSend",
				  KFVM_D_DECL(ps.nX + 2*ps.rad,ps.rad,ps.nZ));
      sFaceSend = FaceDataCommView("Solver::BoundaryData::sFaceSend",
				  KFVM_DM_DECL(ps.nX,ps.nZ));
    }
    
    if (ps.layoutMPI.sSrc != MPI_PROC_NULL) {
      sCellRecv = ConsDataCommView("Solver::BoundaryData::sCellRecv",
				  KFVM_D_DECL(ps.nX + 2*ps.rad,ps.rad,ps.nZ));
      sFaceRecv = FaceDataCommView("Solver::BoundaryData::sFaceRecv",
				  KFVM_DM_DECL(ps.nX,ps.nZ));
    }
    
    if (ps.layoutMPI.nDst != MPI_PROC_NULL) {
      nCellSend = ConsDataCommView("Solver::BoundaryData::nCellSend",
				  KFVM_D_DECL(ps.nX + 2*ps.rad,ps.rad,ps.nZ));
      nFaceSend = FaceDataCommView("Solver::BoundaryData::nFaceSend",
				  KFVM_DM_DECL(ps.nX,ps.nZ));
    }
    
    if (ps.layoutMPI.nSrc != MPI_PROC_NULL) {
      nCellRecv = ConsDataCommView("Solver::BoundaryData::nCellRecv",
				  KFVM_D_DECL(ps.nX + 2*ps.rad,ps.rad,ps.nZ));
      nFaceRecv = FaceDataCommView("Solver::BoundaryData::nFaceRecv",
				  KFVM_DM_DECL(ps.nX,ps.nZ));
    }
    
    if (ps.layoutMPI.bDst != MPI_PROC_NULL) {
      bCellSend = ConsDataCommView("Solver::BoundaryData::bCellSend",
				  KFVM_D_DECL(ps.nX + 2*ps.rad,ps.nY + 2*ps.rad,ps.rad));
      bFaceSend = FaceDataCommView("Solver::BoundaryData::bFaceSend",
				  KFVM_DM_DECL(ps.nX,ps.nY));
    }
    
    if (ps.layoutMPI.bSrc != MPI_PROC_NULL) {
      bCellRecv = ConsDataCommView("Solver::BoundaryData::bCellRecv",
				  KFVM_D_DECL(ps.nX + 2*ps.rad,ps.nY + 2*ps.rad,ps.rad));
      bFaceRecv = FaceDataCommView("Solver::BoundaryData::bFaceRecv",
				  KFVM_DM_DECL(ps.nX,ps.nY));
    }
    
    if (ps.layoutMPI.tDst != MPI_PROC_NULL) {
      tCellSend = ConsDataCommView("Solver::BoundaryData::tCellSend",
				  KFVM_D_DECL(ps.nX + 2*ps.rad,ps.nY + 2*ps.rad,ps.rad));
      tFaceSend = FaceDataCommView("Solver::BoundaryData::tFaceSend",
				  KFVM_DM_DECL(ps.nX,ps.nY));
    }
    
    if (ps.layoutMPI.tSrc != MPI_PROC_NULL) {
      tCellRecv = ConsDataCommView("Solver::BoundaryData::tCellRecv",
				  KFVM_D_DECL(ps.nX + 2*ps.rad,ps.nY + 2*ps.rad,ps.rad));
      tFaceRecv = FaceDataCommView("Solver::BoundaryData::tFaceRecv",
				  KFVM_DM_DECL(ps.nX,ps.nY));
    }
  }
  
} // end namespace KFVM
