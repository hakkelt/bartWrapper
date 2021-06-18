package io.github.hakkelt.bartwrapper;

import static org.junit.Assert.assertThrows;
import static org.junit.jupiter.api.Assertions.assertDoesNotThrow;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.stream.IntStream;

import org.apache.commons.math3.complex.Complex;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

import io.github.hakkelt.ndarrays.BartDimsEnum;
import io.github.hakkelt.ndarrays.ComplexF32NDArray;
import io.github.hakkelt.ndarrays.NDArray;
import io.github.hakkelt.ndarrays.NDArrayCollectors;

class TestBart {
    static Bart bart;
    static NDArray<Complex> array;

    @BeforeAll
    static void setup() throws IOException {
        bart = Bart.getInstance();
        array = new ComplexF32NDArray(3, 128);
        NDArray<Float> increasingNumbers = IntStream.range(-64, 64).boxed().collect(NDArrayCollectors.toRealF32NDArray(128));
        array.slice(0,":").set(increasingNumbers);
        array.slice(1,":").set(increasingNumbers);
        array.slice(2,":").fill(0);
    }

    @Test
    void testDoNothing() {

    }
    
    @Test
    void testRead() throws BartException {
        assertEquals("0 1 2", bart.read("bitmask", "-b", 7));
    }

    @Test
    void testReadArrayInput() throws BartException {
        assertEquals("128 128 1", bart.read("estdims", array));
    }

    @Test
    void testReadError() throws BartException {
        Exception exception = assertThrows(BartException.class, () -> bart.read("cabs", "asdf"));
        assertEquals(Bart.ERROR_BART_UNSUCCESSFUL, exception.getMessage());
    }
    
    @Test
    void testExecute() throws BartException {
        assertDoesNotThrow(() -> bart.execute("bitmask", "-b", 7));
    }

    @Test
    void testExecuteArrayInput() throws BartException {
        assertDoesNotThrow(() -> bart.execute("estdims", array));
    }

    @Test
    void testExecuteError() throws BartException {
        Exception exception = assertThrows(BartException.class, () -> bart.execute("cabs", "asdf"));
        assertEquals(Bart.ERROR_BART_UNSUCCESSFUL, exception.getMessage());
    }

    @Test
    void testRun() throws BartException {
        NDArray<Double> javaAbs = array.abs();
        NDArray<Complex> bartAbs = bart.run("cabs", array).squeeze();
        assertTrue(bartAbs.subtract(javaAbs).abs().sum() / array.length() < 1e-7);
    }

    @Test
    void testRunError() throws BartException {
        Exception exception = assertThrows(BartException.class, () -> bart.run("cabs", "asdf"));
        assertEquals(Bart.ERROR_BART_UNSUCCESSFUL, exception.getMessage());
    }

    @Test
    void testBartDims() throws BartException {
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
    void testRegisterLoadUnregister() throws BartException {
        assertDoesNotThrow(() -> bart.registerMemory("input.mem", array));
        assertDoesNotThrow(() -> bart.registerOutput("output.mem"));
        assertFalse(bart.isMemoryAssociated("output.mem"));
        assertDoesNotThrow(() -> bart.execute("cabs", "input.mem", "output.mem"));
        assertTrue(bart.isMemoryAssociated("input.mem"));
        assertTrue(bart.isMemoryAssociated("output.mem"));
        List<NDArray<Complex>> result = new ArrayList<NDArray<Complex>>();
        assertDoesNotThrow(() -> result.add(bart.loadMemory("output.mem")));
        assertDoesNotThrow(() -> bart.unregisterMemory("input.mem"));
        assertDoesNotThrow(() -> bart.unregisterMemory("output.mem"));
        assertEquals(array.abs(), result.get(0).squeeze().real());
    }

    @Test
    void testRunChained() throws BartException {
        NDArray<Complex> image = bart.run("phantom", "-3").squeeze();
        image.divideInplace(image.norm(Double.POSITIVE_INFINITY));
        NDArray<Complex> kspaceFull = bart.run("fft", 7, image).squeeze();
        NDArray<Complex> image_zf = bart.run("fft", "-i", 7, kspaceFull).squeeze();
        image_zf.divideInplace(image_zf.norm(Double.POSITIVE_INFINITY));
        NDArray<Complex> diff = image.subtract(image_zf);
        assertTrue(diff.norm(Double.POSITIVE_INFINITY) < 1e-6);
    }
}
