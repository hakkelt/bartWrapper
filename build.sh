#!/bin/bash

# Uncomment the following line for debugging
#DEBUG=1

if [[ -z "${JAVA_HOME}" ]]; then
    echo "You must set the JAVA_HOME environmental variable"
    exit 1
fi

OS=$(uname -a | rev | cut -d" " -f1 | rev)

if [[ "$OS" == "Msys" ]]; then
    STATIC_LIB_EXT="lib"
    OUTPUT="bart.dll"
    JNI_DIR="win32"
    OS_DEPENDENT_FLAGS="-static -L/mingw64/lib -lopenblas"
elif [[ "$OS" == "GNU/Linux" ]]; then
    STATIC_LIB_EXT="a"
    OUTPUT="libbart.so"
    JNI_DIR="linux"
    OS_DEPENDENT_FLAGS="-llapacke -lblas -lfftw3f_threads"
else
    echo "Not recognized OS: $OS"
    exit 1
fi

if [ -z $DEBUG ]; then
    PARALLEL=1 make -C src/native/bart
else
    PARALLEL=1 DEBUG=1 make -C src/native/bart
fi
src/native/bart/ar_lock.sh rsU src/native/bart/lib/libmain.a src/native/bart/src/bart.o

## Merge static BART libraries into a single archive
# See: https://stackoverflow.com/a/23621751
echo "create bart.a" > bart.mri
for f in `ls src/native/bart/lib/*.a`; do
    echo "addlib $f" >> bart.mri
done
echo "save" >> bart.mri
echo "end" >> bart.mri
ar -M < bart.mri
rm bart.mri
mv bart.a src/main/resources/bart.$STATIC_LIB_EXT

if [ -z $DEBUG ]; then
    DEBUG_SPECIFIC="-O3"
else
    DEBUG_SPECIFIC="-Og -g"
fi

gcc -shared $DEBUG_SPECIFIC -ffast-math -Wmissing-prototypes -std=gnu11 -fopenmp -Wall -Wextra \
    -I"$JAVA_HOME/include" -I"$JAVA_HOME/include/$JNI_DIR" -I"src/native/bart/src" \
    -Wl,--whole-archive -lpthread -Wl,--no-whole-archive -Wl,--allow-multiple-definition \
    -o src/main/resources/$OUTPUT \
    src/native/Bart.c src/main/resources/bart.$STATIC_LIB_EXT \
    $OS_DEPENDENT_FLAGS -L/usr/lib -lfftw3f -lgfortran -lquadmath -lpng -lm -lz -lrt
