/*
=============================================================================

=============================================================================
@File    :   TACSHelpers.cpp
@Date    :   2024/11/12
@Author  :   Alasdair Christison Gray
@Description :
*/

// =============================================================================
// Standard Library Includes
// =============================================================================

// =============================================================================
// Extension Includes
// =============================================================================
#include "TACSHelpers.h"

// =============================================================================
// Function definitions
// =============================================================================
void createTACSAssembler(const char *filename, TACSAssembler *&assembler) {
  // Create the mesh loader object on MPI_COMM_WORLD. The
  // TACSAssembler object will be created on the same comm
  TACSMeshLoader *const mesh = new TACSMeshLoader(MPI_COMM_WORLD);
  mesh->incref();

  // Create the isotropic material class
  const TacsScalar rho = 2700.0;
  const TacsScalar specific_heat = 921.096;
  const TacsScalar E = 70e9;
  const TacsScalar nu = 0.3;
  const TacsScalar ys = 400e6;
  const TacsScalar cte = 24.0e-6;
  const TacsScalar kappa = 230.0;
  TACSMaterialProperties *const props = new TACSMaterialProperties(rho, specific_heat, E, nu, ys, cte, kappa);

  // Create the stiffness object
  TACSPlaneStressConstitutive *const stiff = new TACSPlaneStressConstitutive(props);
  stiff->incref();

  // Create the model class
  TACSLinearElasticity2D *const model = new TACSLinearElasticity2D(stiff, TACS_NONLINEAR_STRAIN);

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
        TACSElement *elem = NULL;

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

      // TODO: Define node reordering here?

      // Now, create the TACSAssembler object
      int vars_per_node = 2;
      assembler = mesh->createTACS(vars_per_node);
      assembler->incref();
      mesh->decref();
    }
  }
  else {
    fprintf(stderr, "File %s does not exist\n", filename);
  }
}

void displacementField(const TacsScalar x, const TacsScalar y, TacsScalar &u, TacsScalar &v) {
  const TacsScalar dispScale = 1e-2;
  u = dispScale * (sin(2.0 * M_PI * x) + sin(2.0 * M_PI * y));
  v = dispScale * (cos(2.0 * M_PI * x) + cos(2.0 * M_PI * y));
}

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

void writeBCSRMatToFile(BCSRMatData *const matData, const char *filename) {
  FILE *fp = fopen(filename, "w");
  if (fp) {
    const int nrows = matData->nrows;
    const int ncols = matData->ncols;
    const int bsize = matData->bsize;
    const int *rowp = matData->rowp;
    const int *cols = matData->cols;
    const double *A = matData->A;
    const int numBlocks = rowp[nrows];
    const int nnz = numBlocks * bsize * bsize;
    fprintf(fp, "%d %d %d\n", nrows * bsize, ncols * bsize, nnz);

    // For each block row
    for (int ii = 0; ii < nrows; ii++) {
      const int rowStart = rowp[ii];
      const int rowEnd = rowp[ii + 1];
      // For each block in that row
      for (int jj = rowStart; jj < rowEnd; jj++) {
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
    }
  }
  else {
    fprintf(stderr, "Failed to open file %s\n", filename);
  }
}

void writeResidualToFile(TACSBVec *res, const char *filename) {
  FILE *fp = fopen(filename, "w");
  if (fp) {
    // Get the residual array
    TacsScalar *res_array;
    res->getArray(&res_array);
    int size;
    res->getSize(&size);

    // Write the residual to the file, one value per line
    for (int ii = 0; ii < size; ii++) {
      fprintf(fp, "% .17g\n", res_array[ii]);
    }
  }
  else {
    fprintf(stderr, "Failed to open file %s\n", filename);
  }
}
