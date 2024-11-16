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
               TACSAssembler *&assembler,
               TACSMeshLoader *&mesh,
               TACSMaterialProperties *&props,
               TACSPlaneStressConstitutive *&stiff,
               TACSLinearElasticity2D *&model,
               TACSElementBasis *&basis,
               TACSElement2D *&elem);

/**
 * @brief Compute the u and v displacements to set at the point (x, y)
 *
 * @param x x coordinate
 * @param y y coordinate
 * @param u x displacement
 * @param v y displacement
 */
void displacementField(const TacsScalar x, const TacsScalar y, TacsScalar &u, TacsScalar &v);

/**
 * @brief Set the TACS displacement values using a displacementField function
 *
 * @param assembler TACS assembler
 * @param displacementField Function that computes the displacements given coordinates
 */
void setAnalyticDisplacements(
    TACSAssembler *assembler,
    void (*const displacementFieldFunc)(const TacsScalar x, const TacsScalar y, TacsScalar &u, TacsScalar &v));

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

void writeBCSRMatToFile(BCSRMatData *const matData, const char *filename);

void writeTACSSolution(TACSAssembler *assembler, const char *filename);
