#ifndef LAYOUTMPI_H_
#define LAYOUTMPI_H_

#include <array>
#include <cstddef>
#include <cstdint>

#include <mpi.h>

#include "Types.H"

namespace KFVM {
  
  struct LayoutMPI
  {
    int mpiSize,mpiRank;

    // Number of blocks in global domain
    // one block per MPI rank
    idx_t nbX,nbY,nbZ;

    // Position of this block in global domain
    idx_t bIdxX,bIdxY,bIdxZ;

    // Extent of this block in index space
    idx_t bxLo,bxUp,byLo,byUp,bzLo,bzUp;

    // Ranks of neighboring blocks
    std::array<int,6> nbrRank;

    LayoutMPI()
    {
      MPI_Comm_size(MPI_COMM_WORLD,&mpiSize);
      MPI_Comm_rank(MPI_COMM_WORLD,&mpiRank);
    }

    bool distributeBlocks(idx_t nbX_,idx_t nbY_,idx_t nbZ_,
			  const std::array<BCType,6>& bcType)
    {
      // Copy in block counts and check that they are valid
      nbX = nbX_; nbY = nbY_; nbZ = nbZ_;
      if (nbX*nbY*nbZ != mpiSize) {
	// Add proper error checking later
	return false;
      }

      // Convert rank to block position
      rank2idx(mpiRank,bIdxX,bIdxY,bIdxZ);
      bxLo = bIdxX*nbX;
      bxUp = (bIdxX + 1)*nbX;
      byLo = bIdxY*nbY;
      byUp = (bIdxY + 1)*nbY;
      bzLo = bIdxZ*nbZ;
      bzUp = (bIdxZ + 1)*nbZ;

      // Find ranks of all neighbors
      if (bIdxX == 0 && bcType[FaceLabel::west] == BCType::periodic) {
	nbrRank[FaceLabel::west] = idx2rank(nbX - 1,bIdxY,bIdxZ);
      } else {
	nbrRank[FaceLabel::west] = idx2rank(bIdxX - 1,bIdxY,bIdxZ);
      }
	
      if (bIdxX == nbX-1 && bcType[FaceLabel::east] == BCType::periodic) {
	nbrRank[FaceLabel::east] = idx2rank(0,bIdxY,bIdxZ);
      } else {
	nbrRank[FaceLabel::east] = idx2rank(bIdxX + 1,bIdxY,bIdxZ);
      }
	
      if (bIdxY == 0 && bcType[FaceLabel::south] == BCType::periodic) {
	nbrRank[FaceLabel::south] = idx2rank(bIdxX,nbY - 1,bIdxZ);
      } else {
	nbrRank[FaceLabel::south] = idx2rank(bIdxX,bIdxY - 1,bIdxZ);
      }
	
      if (bIdxY == nbY-1 && bcType[FaceLabel::north] == BCType::periodic) {
	nbrRank[FaceLabel::north] = idx2rank(bIdxX,0,bIdxZ);
      } else {
	nbrRank[FaceLabel::north] = idx2rank(bIdxX,bIdxY + 1,bIdxZ);
      }
	
      if (bIdxZ == 0 && bcType[FaceLabel::bottom] == BCType::periodic) {
	nbrRank[FaceLabel::bottom] = idx2rank(bIdxX,bIdxY,nbZ - 1);
      } else {
	nbrRank[FaceLabel::bottom] = idx2rank(bIdxX,bIdxY,bIdxZ - 1);
      }
	
      if (bIdxZ == nbZ-1 && bcType[FaceLabel::top] == BCType::periodic) {
	nbrRank[FaceLabel::top] = idx2rank(bIdxX,bIdxY,0);
      } else {
	nbrRank[FaceLabel::top] = idx2rank(bIdxX,bIdxY,bIdxZ + 1);
      }

      return true;
    }

    void rank2idx(int rank,idx_t& x,idx_t& y,idx_t& z)
    {
      idx_t trank = static_cast<idx_t>(rank);
      idx_t xy = trank % (nbX*nbY);

      x = xy % nbX;
      y = (xy - x)/nbX;
      z = (trank - xy)/(nbX*nbY);
    }

    int idx2rank(idx_t x,idx_t y,idx_t z)
    {
      if (x < 0 || y < 0 || z < 0 ||
	  x >= nbX || y >= nbY || z >= nbZ) {
	// block is outside domain
	return -1;
      }
      return static_cast<int>(nbX*nbY*z + nbX*y + x);
    }
  };

} // end namespace KFVM

#endif
