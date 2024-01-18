#ifndef LAYOUTMPI_H_
#define LAYOUTMPI_H_

#include <array>
#include <cstddef>
#include <cstdint>

#include <mpi.h>

#include <Definitions.H>

#include "Types.H"

namespace KFVM {

struct LayoutMPI {
  const MPI_Datatype realType;
  MPI_Comm commWorld;
  int size, rank;

  // Number of blocks in global domain
  // one block per MPI rank
  idx_t nbX, nbY, nbZ;

  // Position of this block in global domain
  idx_t bIdxX, bIdxY, bIdxZ;

  // Extent of this block in index space
  idx_t bxLo, bxUp, byLo, byUp, bzLo, bzUp;

  // Ranks of neighboring blocks
  int wSrc, wDst, eSrc, eDst;
  int sSrc, sDst, nSrc, nDst;
  int bSrc, bDst, tSrc, tDst;

  // Will this rank communicate in each direction
  bool hasEWComm, hasNSComm, hasTBComm;

  LayoutMPI() : realType(MPI_DOUBLE) {
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  }

  bool distributeBlocks(idx_t nbX_, idx_t nbY_, idx_t nbZ_, idx_t nX, idx_t nY, idx_t nZ,
                        const std::array<BCType, 6> &bcType) {
    // Copy in block counts and check that they are valid
    nbX = nbX_;
    nbY = nbY_;
    nbZ = nbZ_;
    if (int(nbX * nbY * nbZ) != size) {
      // Add proper error checking later
      return false;
    }

    // Create cartesian communicator
    // Always 3d regardless of space dimension
    int dims[] = {int(nbX), int(nbY), int(nbZ)};
    int periods[] = {
        bcType[FaceLabel::west] == BCType::periodic ? 1 : 0,
        bcType[FaceLabel::north] == BCType::periodic ? 1 : 0,
        bcType[FaceLabel::bottom] == BCType::periodic ? 1 : 0,
    };
    MPI_Cart_create(MPI_COMM_WORLD, 3, dims, periods, 1, &commWorld);

    // Convert rank to block position
    int bIdx[3];
    MPI_Cart_coords(commWorld, rank, 3, bIdx);
    bIdxX = bIdx[0];
    bIdxY = bIdx[1];
    bIdxZ = bIdx[2];

    // Set extents of this block in global index space
    bxLo = bIdxX * nX;
    bxUp = (bIdxX + 1) * nX;
    byLo = bIdxY * nY;
    byUp = (bIdxY + 1) * nY;
    bzLo = bIdxZ * nZ;
    bzUp = (bIdxZ + 1) * nZ;

    // Find ranks of all neighbors
    MPI_Cart_shift(commWorld, 0, -1, &eSrc, &wDst);
    MPI_Cart_shift(commWorld, 0, 1, &wSrc, &eDst);
    MPI_Cart_shift(commWorld, 1, -1, &nSrc, &sDst);
    MPI_Cart_shift(commWorld, 1, 1, &sSrc, &nDst);
    MPI_Cart_shift(commWorld, 2, -1, &tSrc, &bDst);
    MPI_Cart_shift(commWorld, 2, 1, &bSrc, &tDst);

    // Store whether certain comms happen or not
    hasEWComm = (wSrc != MPI_PROC_NULL || wDst != MPI_PROC_NULL ||
                 eSrc != MPI_PROC_NULL || eDst != MPI_PROC_NULL);
    hasNSComm = (sSrc != MPI_PROC_NULL || sDst != MPI_PROC_NULL ||
                 nSrc != MPI_PROC_NULL || nDst != MPI_PROC_NULL);
    hasTBComm = (bSrc != MPI_PROC_NULL || bDst != MPI_PROC_NULL ||
                 tSrc != MPI_PROC_NULL || tDst != MPI_PROC_NULL);

    return true;
  }
};

} // end namespace KFVM

#endif
