# Java Interface for BART

[![Documentation](https://img.shields.io/badge/Documentation-latest-blue)](https://hakkelt.github.io/bartWrapper/)

The aim of the code inside this directory is to provide an easy way to call BART commands natively from Java.

## Examples

### Initialize an NDArray
Package NDArrays is a separate package, but it is developed specifically to help passing data to BART.

```java
NDArray<Complex> array = new ComplexF32NDArray(3, 128); // 3 x 128 array initialized with complex zeros
NDArray<Float> increasingNumbers = IntStream.range(-64, 64).boxed().collect(NDArrayCollectors.toRealF32NDArray(128)); // 1D array holding integer values from -64 to 64
array.slice(0,":").set(increasingNumbers); // copy content of increasingNumbers to the first row of the array
array.slice(1,":").set(increasingNumbers); // copy content of increasingNumbers to the second row of the array
array.slice(2,":").fill(new Complex(0,1)); // fill the third row of the array with 0 + 1i
```

### High-level approach

```java
bart = Bart.getInstance();

// Use "read" for functions that return string
String dimensions = bart.read("bitmask", "-b", 7); // "0 1 2"

// Use "execute" if no return value is expected
bart.execute("copy", array, "array.ra"); // saves array to a file

// Use "run" if an array is returned. Also, BartDimsEnum helps to handle dimensions more easily.
BartDimsEnum[] dimsOrder = new BartDimsEnum[]{
    BartDimsEnum._00_READ,
    BartDimsEnum._01_PHS1,
    BartDimsEnum._13_SLICE,
    BartDimsEnum._10_TIME
};
array.setBartDims(dimsOrder); // tell the wrapper that it should reshape the array and permute the dimensions before passing it to BART
NDArray<Double> javaAbs = array.abs();
NDArray<Complex> bartAbs = bart.run("cabs", array).squeeze().permuteDims(dimsOrder); // re-arrange the dimensions to the original order
double diff = bartAbs.subtract(javaAbs).abs().sum() / array.length(); // Should be a small number, e.g. ~1e-7

// BartException signals errors within BART execution
try {
    bart.read("cabs", "asdf")
} catch (BartException e) {
    // bart.read("cabs", "asdf") throws BartException because "asdf" file doesn't exists
}
```

### Low-level approach

Illustrates what "run" does behind the scenes: registers input arrays, registers the name of output, executes the commands, fetches the result, and cleans up BART memory space.

```java
bart = Bart.getInstance();

bart.registerMemory("input.mem", array); // tell BART to read values from array when "input.mem" is passed as an input argument
bart.registerOutput("output.mem");       // tell BART that "output.mem" is going to store output values

Boolean inputRegistered = bart.isMemoryAssociated("input.mem"); // should be true
Boolean outputRegistered = bart.isMemoryAssociated("output.mem"); // should be false because no memory is associated with this name yet

bart.execute("cabs", "input.mem", "output.mem");

outputRegistered = bart.isMemoryAssociated("output.mem"); // this time should be true
NDArray<Complex> result = bart.loadMemory("output.mem"); // fetch result data

// Clean up BART memory
bart.unregisterMemory("input.mem");
bart.unregisterMemory("output.mem");
```

## Dependencies

- `io.github.hakkelt.ndarrays` -> NDArray type to handle multi-dimensional arrays easily and pass them to BART
- `org.apache.commons.math3` -> Complex type for complex numbers

## Building JNI binary

### Prepare MSYS2 environment for Windows
MSYS2 is needed to emulate the Linux environment on Windows in order to ease the process of building.
Commands in the "OS-independent Steps" section below should be executed within the MSYS2 shell after setting it up:

- Install MSYS2 (see: https://www.msys2.org/)
- Install MSYS2 packages required by BART: `src/native/bart/msys_setup.sh`

### OS-independent Steps

- BART submodule needs to be initialized and fetched: `git submodule update --init --recursive`
- BART and the JNI driver needs to be compiled by running the following command also within MSYS2 terminal: `./build.sh`
- The Java driver class expects `bart.dll` (Windows), `libbart.so` (Linux) or `libbart.dylib` (Mac) to be in the `src/main/resources` directory of this project as a result of the previous operations.
