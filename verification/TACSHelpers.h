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

void createTACSAssembler(const char *filename, TACSAssembler *&assembler);

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

void writeResidualToFile(TACSBVec *res, const char *filename);

void writeBCSRMatToFile(BCSRMatData *const matData, const char *filename);

void writeTACSSolution(TACSAssembler *assembler, const char *filename);
