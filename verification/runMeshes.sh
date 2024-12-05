projectDir=/nobackup/achris10/GPU-FEM-Project
for meshFile in $projectDir/FixedMeshes/*.bdf; do
    echo Running with mesh file: $meshFile
    ./TestKernels.exe $meshFile
    if [[ $? != 0 ]]; then
        echo "Opening ${meshFile} failed"
        rm -rf core.*
    fi
done
