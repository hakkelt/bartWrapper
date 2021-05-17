# Java Interface for BART

The aim of the code inside this directory is to provide an easy way to call BART commands natively from Java.

## Building

- BART repository need to be cloned into the parent directory: `../bart`
- MSYS2 need to be installed (see: https://www.msys2.org/)
- BART need to be compiled with MSYS2: `cd ../bart; ./msys_setup.sh; make; cd ../bartConnector`
- JNI driver needs to be compiled by running the following command also within MSYS2 terminal: `./build.sh`
- The Java driver class expects `bart.dll` to be in the root directory of this project as a result of the previous operations.
