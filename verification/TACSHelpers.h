/*
=============================================================================

=============================================================================
@File    :   TACSHelpers.h
@Date    :   2024/11/12
@Author  :   Alasdair Christison Gray
@Description :
*/

#pragma once

// =============================================================================
// Standard Library Includes
// =============================================================================

// =============================================================================
// Extension Includes
// =============================================================================
#include "TACSAssembler.h"
#include "TACSElement2D.h"
#include "TACSLinearElasticity.h"
#include "TACSMeshLoader.h"
#include "TACSQuadBasis.h"
#include "TACSToFH5.h"

// =============================================================================
// Global constant definitions
// =============================================================================

// =============================================================================
// Function prototypes
// =============================================================================

void setupTACS(const char *filename,
               const TACSAssembler::OrderingType nodeOrdering,
               TACSAssembler *&assembler,
               TACSMeshLoader *&mesh,
               TACSMaterialProperties *&props,
               TACSPlaneStressConstitutive *&stiff,
               TACSLinearElasticity2D *&model,
               TACSElementBasis *&basis,
               TACSElement2D *&elem) {
  // Create the mesh loader object on MPI_COMM_WORLD. The
  // TACSAssembler object will be created on the same comm
  mesh = new TACSMeshLoader(MPI_COMM_WORLD);
  mesh->incref();

  // Create the isotropic material class
  const TacsScalar rho = 2700.0;
  const TacsScalar specific_heat = 921.096;
  const TacsScalar E = 70e9;
  const TacsScalar nu = 0.3;
  const TacsScalar ys = 400e6;
  const TacsScalar cte = 24.0e-6;
  const TacsScalar kappa = 230.0;
  const TacsScalar t = 1e-2;
  props = new TACSMaterialProperties(rho, specific_heat, E, nu, ys, cte, kappa);
  props->incref();

  // Create the stiffness object
  stiff = new TACSPlaneStressConstitutive(props, t, 0);
  stiff->incref();

  // Create the model class
  model = new TACSLinearElasticity2D(stiff, TACS_LINEAR_STRAIN);
  model->incref();

  // Create the basis
  TACSElementBasis *linear_basis = new TACSLinearQuadBasis();
  TACSElementBasis *quad_basis = new TACSQuadraticQuadBasis();
  TACSElementBasis *cubic_basis = new TACSCubicQuadBasis();
  TACSElementBasis *quartic_basis = new TACSQuarticQuadBasis();

  // Create the element type
  TACSElement2D *linear_element = new TACSElement2D(model, linear_basis);
  TACSElement2D *quad_element = new TACSElement2D(model, quad_basis);
  TACSElement2D *cubic_element = new TACSElement2D(model, cubic_basis);
  TACSElement2D *quartic_element = new TACSElement2D(model, quartic_basis);

  FILE *fp = fopen(filename, "r");
  if (fp) {
    fclose(fp);

    // Scan the BDF file
    int fail = mesh->scanBDFFile(filename);

    if (fail) {
      fprintf(stderr, "Failed to read in the BDF file\n");
    }
    else {
      // Add the elements to the mesh loader class
      for (int i = 0; i < mesh->getNumComponents(); i++) {

        // Get the BDF description of the element
        const char *elem_descript = mesh->getElementDescript(i);
        if (strcmp(elem_descript, "CQUAD4") == 0) {
          elem = linear_element;
        }
        else if (strcmp(elem_descript, "CQUAD") == 0 || strcmp(elem_descript, "CQUAD9") == 0) {
          elem = quad_element;
        }
        else if (strcmp(elem_descript, "CQUAD16") == 0) {
          elem = cubic_element;
        }
        else if (strcmp(elem_descript, "CQUAD25") == 0) {
          elem = quartic_element;
        }

        // Set the element object into the mesh loader class
        if (elem) {
          mesh->setElement(i, elem);
        }
      }
      elem->incref();
      basis = elem->getElementBasis();
      basis->incref();

      // TODO: Define node reordering here?

      // Now, create the TACSAssembler object
      const int vars_per_node = 2;
      assembler = mesh->createTACS(vars_per_node, nodeOrdering);
      assembler->incref();
    }
  }
  else {
    fprintf(stderr, "File %s does not exist\n", filename);
  }
}

/**
 * map[i][j*numNodesPerElement + k] returns the index in the BCSR data array that the block in the local matrix of
 * element i associated with the coupling between nodes j and k should be written to
 */
void generateElementBCSRMap(const int *const connPtr,
                            const int *const conn,
                            const int numElements,
                            const int numNodesPerElement,
                            const BCSRMatData *const matData,
                            int **&bcsrMap) {
  // Allocate memory (I know doing this inside a function is a bad idea but it'll do for now)
  bcsrMap = new int *[numElements];
  for (int ii = 0; ii < numElements; ii++) {
    bcsrMap[ii] = new int[numNodesPerElement * numNodesPerElement];
  }

  const int blockLength = matData->bsize * matData->bsize;

  // Build the map
  for (int ii = 0; ii < numElements; ii++) {
    const int *const elemNodes = &conn[connPtr[ii]];
    for (int iNode = 0; iNode < numNodesPerElement; iNode++) {
      const int blockRowInd = elemNodes[iNode];
      const int rowStart = matData->rowp[blockRowInd];
      const int rowEnd = matData->rowp[blockRowInd + 1];

      for (int jNode = 0; jNode < numNodesPerElement; jNode++) {
        const int blockColInd = elemNodes[jNode];
        // We know which row and column index in the block matrix we're looking for, but we don't know how many other
        // blocks there are in this row of the matrix, so we need to search along the column indices for this row until
        // we find the one corresponding to this jNode.
        bool foundCol = false;
        for (int jj = rowStart; jj < rowEnd; jj++) {
          if (matData->cols[jj] == blockColInd) {
            foundCol = true;
            bcsrMap[ii][iNode * numNodesPerElement + jNode] = jj * blockLength;
#ifndef NDEBUG
            printf("Block associated with node i = %d, node j = %d in element %d starts at index %d in BCSR data\n",
                   iNode,
                   jNode,
                   ii,
                   jj * blockLength);
#endif
            break;
          }
        }
        if (!foundCol) {
          printf("Couldn't find a block for column %d in row %d\n", blockColInd, blockRowInd);
        }
      }
    }
  }
}

/**
 * @brief Compute the u and v displacements to set at the point (x, y)
 *
 * @param x x coordinate
 * @param y y coordinate
 * @param u x displacement
 * @param v y displacement
 */
void displacementField(const TacsScalar x, const TacsScalar y, TacsScalar &u, TacsScalar &v) {
  const TacsScalar dispScale = 1e-2;
  u = dispScale * (sin(2.0 * M_PI * x) + sin(2.0 * M_PI * y));
  v = dispScale * (cos(2.0 * M_PI * x) + cos(2.0 * M_PI * y));
}

/**
 * @brief Set the TACS displacement values using a displacementField function
 *
 * @param assembler TACS assembler
 * @param displacementField Function that computes the displacements given coordinates
 */
void setAnalyticDisplacements(
    TACSAssembler *assembler,
    void (*const displacementFieldFunc)(const TacsScalar x, const TacsScalar y, TacsScalar &u, TacsScalar &v)) {
  // Create a node vector
  TACSBVec *nodeVec = assembler->createNodeVec();
  nodeVec->incref();
  assembler->getNodes(nodeVec);

  // Get the local node locations
  TacsScalar *Xpts = NULL;
  nodeVec->getArray(&Xpts);

  // Create a vector to set the displacements
  TACSBVec *dispVec = assembler->createVec();
  dispVec->incref();

  TacsScalar *disp;
  dispVec->getArray(&disp);

  // Set the displacements to ux = sin(2 pi x) * sin(2 pi y), uy - cos(2 pi x) * cos(2 pi y)
  // TODO: Are node coordinates still 3D for 2D problem?
  for (int i = 0; i < assembler->getNumNodes(); i++) {
    displacementField(Xpts[3 * i], Xpts[3 * i + 1], disp[2 * i], disp[2 * i + 1]);
  }
  assembler->setVariables(dispVec);

  nodeVec->decref();
  dispVec->decref();
}

template <typename T>
void writeArrayToFile(const T array[], const int n, const char *filename) {
  FILE *fp = fopen(filename, "w");
  if (fp) {
    // Write the residual to the file, one value per line
    for (int ii = 0; ii < n; ii++) {
      fprintf(fp, "% .17g\n", array[ii]);
    }
  }
  else {
    fprintf(stderr, "Failed to open file %s\n", filename);
  }
}

void writeBCSRMatToFile(BCSRMatData *const matData,
                        const char *filename,
                        const int maxRows = -1,
                        const int maxCols = -1) {
  FILE *fp = fopen(filename, "w");
  if (fp) {
    int nrows = matData->nrows;
    int ncols = matData->ncols;
    const int bsize = matData->bsize;
    const int *rowp = matData->rowp;
    const int *cols = matData->cols;
    const double *A = matData->A;
    const int numBlocks = rowp[nrows];
    const int bsize2 = bsize * bsize;
    int nnz = numBlocks * bsize2;

    // Possibly limit the number of rows and columns to write
    bool recomputeNNZ = false;
    if (maxRows > 0) {
      nrows = std::min(maxRows, nrows);
      recomputeNNZ = true;
    }
    if (maxCols > 0) {
      ncols = std::min(maxCols, ncols);
      recomputeNNZ = true;
    }

    // If we've limited the number of rows or columns, we need to recompute the number of non-zero blocks
    if (recomputeNNZ) {
      nnz = 0;
      for (int ii = 0; ii < nrows; ii++) {
        const int rowStart = rowp[ii];
        const int rowEnd = rowp[ii + 1];
        for (int jj = rowStart; jj < rowEnd; jj++) {
          if (cols[jj] < ncols) {
            nnz += bsize2;
          }
          else {
            break;
          }
        }
      }
    }

    fprintf(fp, "%d %d %d\n", nrows * bsize, ncols * bsize, nnz);

    // For each block row (up to the max number of rows)
    for (int ii = 0; ii < nrows; ii++) {
      const int rowStart = rowp[ii];
      const int rowEnd = rowp[ii + 1];
      // For each block in that row (up to the max number of columns)
      for (int jj = rowStart; jj < rowEnd; jj++) {
        if (cols[jj] < ncols) {
          const TacsScalar *const block = &A[bsize * bsize * jj];
          // For each row and col in the block
          for (int kk = 0; kk < bsize; kk++) {
            const int globalRow = ii * bsize + kk;
            for (int ll = 0; ll < bsize; ll++) {
              const int globalCol = cols[jj] * bsize + ll;
              fprintf(fp, "%d %d % .17g\n", globalRow, globalCol, block[bsize * kk + ll]);
            }
          }
        }
        else {
          break;
        }
      }
    }
  }
  else {
    fprintf(stderr, "Failed to open file %s\n", filename);
  }
}

void writeTACSSolution(TACSAssembler *assembler, const char *filename) {
  // Create an TACSToFH5 object for writing output to files
  ElementType etype = TACS_PLANE_STRESS_ELEMENT;
  int write_flag = (TACS_OUTPUT_CONNECTIVITY | TACS_OUTPUT_NODES | TACS_OUTPUT_DISPLACEMENTS | TACS_OUTPUT_STRAINS |
                    TACS_OUTPUT_STRESSES | TACS_OUTPUT_EXTRAS);
  TACSToFH5 *f5 = new TACSToFH5(assembler, etype, write_flag);
  f5->incref();
  f5->writeToFile(filename);

  // Free everything
  f5->decref();
}
