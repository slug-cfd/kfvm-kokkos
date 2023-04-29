#include <Kokkos_Core.hpp>

#include "Dimension.H"
#include "Types.H"
#include "ProblemSetup.H"
#include "BoundaryData.H"

namespace KFVM {

  BoundaryData::BoundaryData(const ProblemSetup& ps):
    wCellSrcLo({KFVM_D_DECL(ps.rad  ,ps.rad        ,ps.rad        )}),
    wCellSrcUp({KFVM_D_DECL(2*ps.rad,ps.nY + ps.rad,ps.nZ + ps.rad)}),
    wCellDstLo({KFVM_D_DECL(0       ,ps.rad        ,ps.rad        )}),
    wCellDstUp({KFVM_D_DECL(ps.rad  ,ps.nY + ps.rad,ps.nZ + ps.rad)}),
    eCellSrcLo({KFVM_D_DECL(ps.nX           ,ps.rad        ,ps.rad        )}),
    eCellSrcUp({KFVM_D_DECL(ps.nX + ps.rad  ,ps.nY + ps.rad,ps.nZ + ps.rad)}),
    eCellDstLo({KFVM_D_DECL(ps.nX + ps.rad  ,ps.rad        ,ps.rad        )}),
    eCellDstUp({KFVM_D_DECL(ps.nX + 2*ps.rad,ps.nY + ps.rad,ps.nZ + ps.rad)}),
    sCellSrcLo({KFVM_D_DECL(0               ,ps.rad  ,ps.rad        )}),
    sCellSrcUp({KFVM_D_DECL(ps.nX + 2*ps.rad,2*ps.rad,ps.nZ + ps.rad)}),
    sCellDstLo({KFVM_D_DECL(0               ,0       ,ps.rad        )}),
    sCellDstUp({KFVM_D_DECL(ps.nX + 2*ps.rad,ps.rad  ,ps.nZ + ps.rad)}),
    nCellSrcLo({KFVM_D_DECL(0               ,ps.nY           ,ps.rad        )}),
    nCellSrcUp({KFVM_D_DECL(ps.nX + 2*ps.rad,ps.nY + ps.rad  ,ps.nZ + ps.rad)}),
    nCellDstLo({KFVM_D_DECL(0               ,ps.nY + ps.rad  ,ps.rad        )}),
    nCellDstUp({KFVM_D_DECL(ps.nX + 2*ps.rad,ps.nY + 2*ps.rad,ps.nZ + ps.rad)}),
    bCellSrcLo({KFVM_D_DECL(0               ,0               ,ps.rad  )}),
    bCellSrcUp({KFVM_D_DECL(ps.nX + 2*ps.rad,ps.nY + 2*ps.rad,2*ps.rad)}),
    bCellDstLo({KFVM_D_DECL(0               ,0               ,0       )}),
    bCellDstUp({KFVM_D_DECL(ps.nX + 2*ps.rad,ps.nY + 2*ps.rad,ps.rad  )}),
    tCellSrcLo({KFVM_D_DECL(0               ,0               ,ps.nZ           )}),
    tCellSrcUp({KFVM_D_DECL(ps.nX + 2*ps.rad,ps.nY + 2*ps.rad,ps.nZ + ps.rad  )}),
    tCellDstLo({KFVM_D_DECL(0               ,0               ,ps.nZ + ps.rad  )}),
    tCellDstUp({KFVM_D_DECL(ps.nX + 2*ps.rad,ps.nY + 2*ps.rad,ps.nZ + 2*ps.rad)})
  {
    // Allocate and assign views as needed
    if (ps.layoutMPI.nbrRank[FaceLabel::west] >=0 &&
	ps.layoutMPI.nbrRank[FaceLabel::west] != ps.layoutMPI.nbrRank[FaceLabel::east]) {
      wCellSrc = ConsDataCommView("Solver::BoundaryData::wCellSrc",
				  KFVM_D_DECL(ps.rad,ps.nY,ps.nZ));
      wCellDst = ConsDataCommView("Solver::BoundaryData::wCellDst",
				  KFVM_D_DECL(ps.rad,ps.nY,ps.nZ));
      wFaceSrc = FaceDataCommView("Solver::BoundaryData::wFaceSrc",
				  KFVM_DM_DECL(ps.nY,ps.nZ));
      wFaceDst = FaceDataCommView("Solver::BoundaryData::wFaceDst",
				  KFVM_DM_DECL(ps.nY,ps.nZ));
    }
    
    if (ps.layoutMPI.nbrRank[FaceLabel::east] >=0 &&
	ps.layoutMPI.nbrRank[FaceLabel::west] != ps.layoutMPI.nbrRank[FaceLabel::east]) {
      eCellSrc = ConsDataCommView("Solver::BoundaryData::eCellSrc",
				  KFVM_D_DECL(ps.rad,ps.nY,ps.nZ));
      eCellDst = ConsDataCommView("Solver::BoundaryData::eCellDst",
				  KFVM_D_DECL(ps.rad,ps.nY,ps.nZ));
      eFaceSrc = FaceDataCommView("Solver::BoundaryData::eFaceSrc",
				  KFVM_DM_DECL(ps.nY,ps.nZ));
      eFaceDst = FaceDataCommView("Solver::BoundaryData::eFaceDst",
				  KFVM_DM_DECL(ps.nY,ps.nZ));
    }
    
    if (ps.layoutMPI.nbrRank[FaceLabel::south] >=0 &&
	ps.layoutMPI.nbrRank[FaceLabel::south] != ps.layoutMPI.nbrRank[FaceLabel::north]) {
      sCellSrc = ConsDataCommView("Solver::BoundaryData::sCellSrc",
				  KFVM_D_DECL(ps.nX + 2*ps.rad,ps.rad,ps.nZ));
      sCellDst = ConsDataCommView("Solver::BoundaryData::sCellDst",
				  KFVM_D_DECL(ps.nX + 2*ps.rad,ps.rad,ps.nZ));
      sFaceSrc = FaceDataCommView("Solver::BoundaryData::sFaceSrc",
				  KFVM_DM_DECL(ps.nX,ps.nZ));
      sFaceDst = FaceDataCommView("Solver::BoundaryData::sFaceDst",
				  KFVM_DM_DECL(ps.nX,ps.nZ));
    }
    
    if (ps.layoutMPI.nbrRank[FaceLabel::north] >=0 &&
	ps.layoutMPI.nbrRank[FaceLabel::south] != ps.layoutMPI.nbrRank[FaceLabel::north]) {
      nCellSrc = ConsDataCommView("Solver::BoundaryData::nCellSrc",
				  KFVM_D_DECL(ps.nX + 2*ps.rad,ps.rad,ps.nZ));
      nCellDst = ConsDataCommView("Solver::BoundaryData::nCellDst",
				  KFVM_D_DECL(ps.nX + 2*ps.rad,ps.rad,ps.nZ));
      nFaceSrc = FaceDataCommView("Solver::BoundaryData::nFaceSrc",
				  KFVM_DM_DECL(ps.nX,ps.nZ));
      nFaceDst = FaceDataCommView("Solver::BoundaryData::nFaceDst",
				  KFVM_DM_DECL(ps.nX,ps.nZ));
    }
    
    if (ps.layoutMPI.nbrRank[FaceLabel::bottom] >=0 &&
	ps.layoutMPI.nbrRank[FaceLabel::bottom] != ps.layoutMPI.nbrRank[FaceLabel::top]) {
      bCellSrc = ConsDataCommView("Solver::BoundaryData::bCellSrc",
				  KFVM_D_DECL(ps.nX + 2*ps.rad,ps.nY + 2*ps.rad,ps.rad));
      bCellDst = ConsDataCommView("Solver::BoundaryData::bCellDst",
				  KFVM_D_DECL(ps.nX + 2*ps.rad,ps.nY + 2*ps.rad,ps.rad));
      bFaceSrc = FaceDataCommView("Solver::BoundaryData::bFaceSrc",
				  KFVM_DM_DECL(ps.nX,ps.nY));
      bFaceDst = FaceDataCommView("Solver::BoundaryData::bFaceDst",
				  KFVM_DM_DECL(ps.nX,ps.nY));
    }
    
    if (ps.layoutMPI.nbrRank[FaceLabel::top] >=0 &&
	ps.layoutMPI.nbrRank[FaceLabel::bottom] != ps.layoutMPI.nbrRank[FaceLabel::top]) {
      tCellSrc = ConsDataCommView("Solver::BoundaryData::tCellSrc",
				  KFVM_D_DECL(ps.nX + 2*ps.rad,ps.nY + 2*ps.rad,ps.rad));
      tCellDst = ConsDataCommView("Solver::BoundaryData::tCellDst",
				  KFVM_D_DECL(ps.nX + 2*ps.rad,ps.nY + 2*ps.rad,ps.rad));
      tFaceSrc = FaceDataCommView("Solver::BoundaryData::tFaceSrc",
				  KFVM_DM_DECL(ps.nX,ps.nY));
      tFaceDst = FaceDataCommView("Solver::BoundaryData::tFaceDst",
				  KFVM_DM_DECL(ps.nX,ps.nY));
    }
  }
  
} // end namespace KFVM
