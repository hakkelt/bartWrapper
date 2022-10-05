package io.github.hakkelt.bartwrapper;

import static org.junit.Assert.assertThrows;
import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertDoesNotThrow;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.io.File;
import java.io.IOException;
import java.util.stream.IntStream;

import org.apache.commons.math3.complex.Complex;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

import io.github.hakkelt.ndarrays.NDArray;
import io.github.hakkelt.ndarrays.basic.BasicFloatNDArray;

class TestBart {
    static BartNDArray array;

    @BeforeAll
    static void setup() throws IOException {
        array = new BartComplexFloatNDArray(3, 128);
        NDArray<Float> increasingNumbers = IntStream.range(-64, 64).boxed().collect(BasicFloatNDArray.getCollector(128));
        array.slice(0,":").copyFrom(increasingNumbers);
        array.slice(1,":").copyFrom(increasingNumbers);
        array.slice(2,":").fill(0);
    }
    
    @Test
    void testRead() throws BartException {
        assertEquals("0 1 2", Bart.read("bitmask", "-b", 7));
    }

    @Test
    void testReadArrayInput() throws BartException {
        assertEquals("128 128 1", Bart.read("estdims", array));
    }

    @Test
    void testReadError() throws BartException {
        Exception exception = assertThrows(BartException.class, () -> Bart.read("cabs", "asdf"));
        assertEquals(String.format(BartErrors.BART_UNSUCCESSFUL,
            "Usage: cabs <input> <output> " + System.lineSeparator() +
            "ERROR: cmdline: too few or too many arguments"),
            exception.getMessage());
    }
    
    @Test
    void testExecute() throws BartException {
        assertDoesNotThrow(() -> Bart.execute("bitmask", "-b", 7));
    }
    
    @Test
    void testExecuteWithOutputConsumer() throws BartException {
        StringBuilder str = new StringBuilder();
        Bart.execute(line -> str.append(line), "bitmask", "-b", 7);
        assertEquals("0 1 2", str.toString());
    }

    @Test
    void testExecuteArrayInput() throws BartException, InterruptedException, IOException {
        assertDoesNotThrow(() -> Bart.execute("estdims", array));
    }

    @Test
    void testExecuteError() throws BartException {
        Exception exception = assertThrows(BartException.class, () -> Bart.execute("cabs", "asdf"));
        assertEquals(String.format(BartErrors.BART_UNSUCCESSFUL, 
            "Usage: cabs <input> <output> " + System.lineSeparator() +
            "ERROR: cmdline: too few or too many arguments"),
            exception.getMessage());
    }

    @Test
    void testRun() throws BartException {
        NDArray<Float> javaAbs = array.abs();
        NDArray<Complex> bartAbs = Bart.run("cabs", array).squeeze();
        javaAbs.forEachWithLinearIndices((value, i) -> assertEquals((double)value, bartAbs.get(i).abs()));
    }

    @Test
    void testRunError() throws BartException {
        Exception exception = assertThrows(BartException.class, () -> Bart.run("cabs", "asdf"));
        assertEquals(String.format(BartErrors.BART_UNSUCCESSFUL,
            "Loading cfl file asdf" + System.lineSeparator() +
            " : No such file or directory" + System.lineSeparator() +
            "ERROR:"), exception.getMessage());
    }

    @Test
    void testUninitializedBartDims() {
        BartNDArray array = new BartComplexFloatNDArray(128, 128, 20, 5);
        Exception exception = assertThrows(UnsupportedOperationException.class, () -> array.getBartDims());
        assertEquals(BartErrors.UNINITIALIZED_BART_DIMS, exception.getMessage());
    }

    @Test
    void testSetBartDims() throws BartException {
        BartDimsEnum[] shapeOrder = new BartDimsEnum[]{
            BartDimsEnum._00_READ,
            BartDimsEnum._01_PHS1,
            BartDimsEnum._13_SLICE,
            BartDimsEnum._10_TIME
        };
        BartNDArray array = new BartComplexFloatNDArray(10, 10, 2, 2)
            .fillUsingLinearIndices(i -> new Complex(i,-i));
        array.setBartDims(shapeOrder);
        assertArrayEquals(shapeOrder, array.getBartDims());
        NDArray<Float> javaAbs = array.abs();
        NDArray<Complex> bartAbs = Bart.run("cabs", array).squeeze().permuteDims(0,1,3,2);
        javaAbs.forEachWithLinearIndices((value, i) -> assertEquals((double) value, bartAbs.get(i).abs()));
    }

    @Test
    void testCopyConstructorWithBartDims() {
        array.setBartDims(BartDimsEnum._00_READ, BartDimsEnum._01_PHS1);
        BartNDArray array2 = new BartComplexFloatNDArray(array);
        assertTrue(array2.areBartDimsSpecified());
        assertArrayEquals(array.getBartDims(), array2.getBartDims());
        array.setBartDims(BartDimsEnum._00_READ, BartDimsEnum._03_COIL);
        assertNotEquals(array.getBartDims()[1], array2.getBartDims()[1]);
    }

    @Test
    void testSetBartDimsSizeMismatch() {
        BartDimsEnum[] shapeOrder = new BartDimsEnum[]{
            BartDimsEnum._00_READ,
            BartDimsEnum._01_PHS1,
            BartDimsEnum._13_SLICE
        };
        BartNDArray array = new BartComplexFloatNDArray(128, 128, 20, 5);
        Exception exception = assertThrows(IllegalArgumentException.class, () -> array.setBartDims(shapeOrder));
        assertEquals(String.format(BartErrors.SET_BART_DIMS_SIZE_MISMATCH, array.ndim()), exception.getMessage());
    }

    @Test
    void testSetBartDimsDuplicates() {
        BartDimsEnum[] shapeOrder = new BartDimsEnum[]{
            BartDimsEnum._00_READ,
            BartDimsEnum._13_SLICE,
            BartDimsEnum._01_PHS1,
            BartDimsEnum._13_SLICE
        };
        BartNDArray array = new BartComplexFloatNDArray(128, 128, 20, 5);
        Exception exception = assertThrows(IllegalArgumentException.class, () -> array.setBartDims(shapeOrder));
        assertEquals(BartErrors.SET_BART_DIMS_DUPLICATES, exception.getMessage());
    }

    @Test
    void testSelectAndReorderBartDims() {
        BartNDArray array = new BartComplexFloatNDArray(128, 1, 64).fill(new Complex(1,-1));
        array.setBartDims(BartDimsEnum._01_PHS1, BartDimsEnum._00_READ, BartDimsEnum._10_TIME);
        BartNDArray array2 = array.selectAndReorderBartDims(BartDimsEnum._10_TIME, BartDimsEnum._01_PHS1);
        assertEquals(2, array2.ndim());
        assertEquals(64, array2.shape(0));
        assertEquals(128, array2.shape(1));
    }

    @Test
    void testSelectAndReorderBartDimsNoSelection() {
        BartNDArray array = new BartComplexFloatNDArray(128, 1, 64).fill(new Complex(1,-1));
        array.setBartDims(BartDimsEnum._01_PHS1, BartDimsEnum._00_READ, BartDimsEnum._10_TIME);
        BartNDArray array2 = array.selectAndReorderBartDims(
            BartDimsEnum._10_TIME, BartDimsEnum._01_PHS1, BartDimsEnum._00_READ);
        assertEquals(3, array2.ndim());
        assertArrayEquals(new int[]{ 64, 128, 1 }, array2.shape());
    }

    @Test
    void testSelectAndReorderBartDimsNoReordering() {
        BartNDArray array = new BartComplexFloatNDArray(128, 1, 64).fill(new Complex(1,-1));
        array.setBartDims(BartDimsEnum._01_PHS1, BartDimsEnum._00_READ, BartDimsEnum._10_TIME);
        BartNDArray array2 = array.selectAndReorderBartDims(BartDimsEnum._01_PHS1, BartDimsEnum._10_TIME);
        assertEquals(2, array2.ndim());
        assertEquals(128, array2.shape(0));
        assertEquals(64, array2.shape(1));
    }

    @Test
    void testRegisterMemoryNotRaExtension() throws BartException {
        File file = new File("input.memo");
        Exception exception = assertThrows(IllegalArgumentException.class, () -> BartNDArray.save(array, file));
        assertEquals(BartErrors.NAME_EXTENSION_IS_NOT_RA, exception.getMessage());
    }

    @Test
    void testRunChained() throws BartException {
        NDArray<Complex> image = new BartComplexFloatNDArray(30, 30)
            .fillUsingCartesianIndices(idx -> new Complex(Math.abs(idx[0] - 15) < 8 && Math.abs(idx[1] - 15) < 8 ? 1 : 0));
        image.divideInplace(image.norm(Double.POSITIVE_INFINITY));
        NDArray<Complex> kspaceFull = Bart.run("fft", 7, image).squeeze();
        NDArray<Complex> image_zf = Bart.run("fft", "-i", 7, kspaceFull).squeeze();
        image_zf.divideInplace(image_zf.norm(Double.POSITIVE_INFINITY));
        NDArray<Complex> diff = image.subtract(image_zf);
        assertEquals(0, diff.norm(Double.POSITIVE_INFINITY), 1e-6);
    }

    @Test
    void testFloatDoubleInput() throws BartException {
        NDArray<Complex> output = Bart.run("vec", 0.5, 0.25f).squeeze();
        assertEquals(0.5, output.get(0).getReal());
        assertEquals(0.25, output.get(1).getReal());
    }

    @Test
    void testUnsupportedType() throws BartException {
        Exception exception = assertThrows(IllegalArgumentException.class, () -> Bart.run("vec", 0.5, 0.25f, BartDimsEnum._00_READ));
        assertEquals(String.format(
                        BartErrors.INPUT_UNSUPPORTED_TYPE,
                        BartDimsEnum._00_READ.toString(),
                        BartDimsEnum._00_READ.getClass()
                    ),
                     exception.getMessage());
    }

}
