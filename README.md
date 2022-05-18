# Java Interface for BART

[![Documentation](https://img.shields.io/badge/Documentation-BartWrapper-blue)](https://hakkelt.github.io/bartWrapper/)
[![Documentation](https://img.shields.io/badge/Documentation-NDArrays-blue)](https://hakkelt.github.io/NDArrays/)
[![BART home](https://img.shields.io/badge/Website-BART-green)](https://hakkelt.github.io/bartWrapper/)
[![BART home](https://img.shields.io/badge/GitHub-BART-black)](https://github.com/mrirecon/bart)

The aim of the code inside this directory is to provide an easy way to call BART commands from Java.

## Examples

### Initialize an NDArray
Package NDArrays is a separate package that aims to provide a general framework to work with multidimensional (complex) arrays.

```java
ComplexNDArray<Float> array = new BartComplexFloatNDArray(3, 128); // BartComplexFloatNDArray is an implementation of the ComplexNDArray interface
array.slice("0:1",":").fillUsingCartesianIndices(idx -> new Complex(idx[1] - 64)); // fill first two rows with increasing integers from -64 to 63
array.slice(2, ":").fill(0); // fill the third row of the array with zeros
```

### High-level approach

```java
// Use "read" for functions that return string
String dimensions = Bart.read("bitmask", "-b", 7);
assertEquals("0 1 2", dimensions);

// Use "execute" if no return value is expected
Bart.execute("copy", array, "array.ra"); // saves array to a file

// Use "run" if an array is returned. Also, BartDimsEnum helps to handle BART dimensions (see: https://github.com/mrirecon/bart/blob/master/README section 3.2) more easily.
BartDimsEnum[] dimsOrder = new BartDimsEnum[]{
    BartDimsEnum._01_PHS1, // First phase encoding dimension
    BartDimsEnum._00_READ  // "Readout" or frequency encoding dimension
};
array.setBartDims(dimsOrder); // tell the wrapper that it should reshape the array and permute the dimensions before passing it to BART
NDArray<Float> javaAbs = array.abs();
ComplexNDArray<Float> bartAbs = Bart.run("cabs", array).selectAndReorderBartDims(dimsOrder); // re-arrange the dimensions to the original order
assertEquals(0, bartAbs.subtract(javaAbs).abs().sum() / array.length(), 1e-7);

// BartException signals errors within BART execution
try {
    Bart.run("cabs", "asdf");
} catch (BartException e) {
    // BartException is thrown because "asdf" file doesn't exists
    System.out.println("Error handled");
}
```

### Low-level approach

This section illustrates what "run" does behind the scenes:
 - saves input arrays to temp,
 - generates name for output,
 - executes the command,
 - reads the result, and
 - cleans up temp files.

```java
File input = BartNDArray.saveToTemp(array);
String inputFileName = input.getName();

File output = Files.createTempFile("bart_", ".ra").toFile();
String outputFileName = output.getName();

Bart.execute("bart", "cabs", inputFileName, outputFileName);

BartNDArray result = BartNDArray.load(output);
result.setBartDims(Stream.of(BartDimsEnum.values()).limit(result.ndim()).toArray(BartDimsEnum[]::new));

Files.delete(input.toPath());
Files.delete(output.toPath());
```

## Dependencies

- `io.github.hakkelt.ndarrays` -> NDArray type to handle multi-dimensional arrays easily and pass them to BART
- `org.apache.commons.math3` -> Complex type for complex numbers

## How to Build the Package

### Preparations:
 - **Windows:**
   - MSYS2 is needed to emulate the Linux environment on Windows in order to ease the process of building. Commands in the "OS-independent Steps" section below should be executed within the MSYS2 shell after setting it up:
   - Install MSYS2 (see: https://www.msys2.org/)
   - Install MSYS2 packages required by BART: `src/native/bart/msys_setup.sh`
 - **Linux:**
   - Dependencies you should install (most likely via the distro's package manager, like `apt` in Ubuntu): `make gcc libfftw3-dev liblapacke-dev libpng-dev libopenblas-dev`

### OS-independent Steps

- BART submodule needs to be initialized and fetched: `git submodule update --init --recursive`
- Complile BART: `PARALLEL=1 make -C src/native/bart` (if you encounter any errors during compilation, try `PARALLEL=0 make -C src/native/bart`)
- Copy compiled binary (`bart.exe` (Windows), `bart` (Linux and Mac)) to `src/main/resources`.
- Build a jar with Maven.
