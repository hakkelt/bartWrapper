# Java Interface for BART

The aim of the code inside this directory is to provide an easy way to call BART commands natively from Java.

## Building

### Prepare MSYS2 environment for Windows
MSYS2 is needed to emulate the Linux environment on Windows in order to ease the process of building.
Commands in the "step" section should be executed within the MSYS2 shell after setting it up:

- Install MSYS2 (see: https://www.msys2.org/)
- Install MSYS2 packages required by BART: `src/native/bart/msys_setup.sh`

### Step

- BART submodule needs to be initialized and fetched: `git submodule update --init --recursive`
- BART and the JNI driver needs to be compiled by running the following command also within MSYS2 terminal: `./build.sh`
- The Java driver class expects `bart.dll` (Windows), `libbart.so` (Linux) or `libbart.dylib` (Mac) to be in the `src/main/resources` directory of this project as a result of the previous operations.
