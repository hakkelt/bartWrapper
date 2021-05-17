package com.mediso.mri.utils;

import static org.junit.Assert.assertThrows;
import static org.junit.jupiter.api.Assertions.assertDoesNotThrow;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.ArrayList;
import java.util.List;
import java.util.stream.IntStream;

import org.apache.commons.math3.complex.Complex;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

public class TestBartConnector {
    static BartConnector bart = new BartConnector();
    static NDArray<Complex> array;

    @BeforeAll
    public static void setup() {
        array = new ComplexF32NDArray(3, 128);
        NDArray<Float> increasingNumbers = IntStream.range(-64, 64).boxed().collect(NDArrayCollectors.toRealF32NDArray(128));
        array.slice(0,":").copyFrom(increasingNumbers);
        array.slice(1,":").copyFrom(increasingNumbers);
        array.slice(2,":").fill(0);
    }
    
    @Test
    public void testRead() throws BartException {
        assertEquals("0 1 2", bart.read("bitmask", "-b", 7));
    }

    @Test
    public void testReadArrayInput() throws BartException {
        assertEquals("128 128 1", bart.read("estdims", array));
    }

    @Test
    public void testReadError() throws BartException {
        Exception exception = assertThrows(BartException.class, () -> bart.read("cabs"));
        assertEquals(BartConnector.ERROR_BART_UNSUCCESSFUL, exception.getMessage());
    }
    
    @Test
    public void testExecute() throws BartException {
        assertDoesNotThrow(() -> bart.execute("bitmask", "-b", 7));
    }

    @Test
    public void testExecuteArrayInput() throws BartException {
        assertDoesNotThrow(() -> bart.execute("estdims", array));
    }

    @Test
    public void testExecuteError() throws BartException {
        Exception exception = assertThrows(BartException.class, () -> bart.execute("cabs"));
        assertEquals(BartConnector.ERROR_BART_UNSUCCESSFUL, exception.getMessage());
    }

    @Test
    public void testRun() throws BartException {
        NDArray<Double> javaAbs = array.abs();
        NDArray<Complex> bartAbs = bart.run("cabs", array).squeeze();
        assertTrue(bartAbs.subtract(javaAbs).abs().sum() / array.length() < 1e-7);
    }

    @Test
    public void testRunError() throws BartException {
        Exception exception = assertThrows(BartException.class, () -> bart.run("cabs"));
        assertEquals(BartConnector.ERROR_BART_UNSUCCESSFUL, exception.getMessage());
    }

    @Test
    public void testBartDims() throws BartException {
        BartDimsEnum[] dimsOrder = new BartDimsEnum[]{
            BartDimsEnum._00_READ,
            BartDimsEnum._01_PHS1,
            BartDimsEnum._13_SLICE,
            BartDimsEnum._10_TIME
        };
        NDArray<Complex> array = IntStream.range(0, 128*128*20*5).parallel()
            .mapToObj(i -> new Complex(i,-i)).collect(NDArrayCollectors.toComplexF32NDArray(128, 128, 20, 5));
        array.setBartDims(dimsOrder);
        NDArray<Double> javaAbs = array.abs();
        NDArray<Complex> bartAbs = bart.run("cabs", array).squeeze().permuteDims(dimsOrder);
        assertTrue(bartAbs.subtract(javaAbs).abs().sum() / array.length() < 1e-7);
    }

    @Test
    public void testRegisterLoad() throws BartException {
        assertDoesNotThrow(() -> bart.registerMemory("input.mem", array));
        assertDoesNotThrow(() -> bart.execute("cabs", "input.mem", "output.mem"));
        List<NDArray<Complex>> result = new ArrayList<NDArray<Complex>>();
        assertDoesNotThrow(() -> result.add(bart.loadResult("output.mem")));
        assertEquals(array.abs(), result.get(0).squeeze().real());
    }
}
