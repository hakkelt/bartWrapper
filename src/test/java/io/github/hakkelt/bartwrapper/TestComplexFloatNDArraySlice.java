package io.github.hakkelt.bartwrapper;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.Arrays;
import java.util.concurrent.atomic.AtomicInteger;

import org.apache.commons.math3.complex.Complex;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import io.github.hakkelt.ndarrays.ComplexNDArray;
import io.github.hakkelt.ndarrays.internal.Errors;
import io.github.hakkelt.ndarrays.NDArray;
import io.github.hakkelt.ndarrays.basic.BasicByteNDArray;
import io.github.hakkelt.ndarrays.basic.BasicComplexFloatNDArray;

class TestComplexFloatNDArraySlice implements NameTrait {
    BartNDArray array, slice;

    @BeforeEach
    void setup() {
        array = new BartComplexFloatNDArray(new int[]{ 4, 5, 3 });
        array.fillUsingLinearIndices(index -> new Complex(index, -index));
        slice = array.slice(1, "1:4", ":");
    }

    @Test
    void testBartDims() {
        assertFalse(slice.areBartDimsSpecified());

        array.setBartDims(BartDimsEnum._00_READ, BartDimsEnum._01_PHS1, BartDimsEnum._02_PHS2);
        slice = array.slice(1, "1:4", ":");
        assertTrue(slice.areBartDimsSpecified());

        assertArrayEquals(new BartDimsEnum[]{
            BartDimsEnum._01_PHS1,
            BartDimsEnum._02_PHS2
        }, slice.getBartDims());

        Exception exception = assertThrows(UnsupportedOperationException.class, () -> slice.setBartDims(BartDimsEnum._00_READ));
        assertEquals(BartErrors.CANNOT_SET_BART_DIMS_ON_VIEW, exception.getMessage());
    }

    @Test
    void testSliceIdentity() {
        BartNDArray slice2 = array.slice(":", "0:5", ":");
        assertSame(array, slice2);
    }

    @Test
    void testSliceSlice() {
        BartNDArray slice2 = slice.slice(":", 2);
        slice2.forEachWithCartesianIndices((value, indices) -> assertEquals(value, array.get(1, indices[0] + 1, 2)));
    }

    @Test
    void testSelectDimsIdentity() {
        BartNDArray slice2 = array.selectDims(2, 0, 1);
        assertSame(array, slice2);
    }

    @Test
    void testDropDims() {
        BartNDArray array2 = new BartComplexFloatNDArray(1, 1, 1);
        Exception exception = assertThrows(IllegalArgumentException.class, () -> array2.dropDims(0, 1, 2));
        assertEquals(Errors.ALL_DIMS_DROPPED, exception.getMessage());
        BartNDArray slice2 = array2.dropDims(2, 0);
        assertEquals(1, slice2.ndim());
    }

    @Test
    void testGetNegativeLinearIndexing() {
        assertEquals(new Complex(45, -45), slice.get(-3));
    }

    @Test
    void testGetNegativeCartesianIndexing() {
        // linearIndex equal to cartesian index [1,3,2] in original array and [2,2] in slice:
        int linearIndex = (2 * 5 + (1 + 2)) * 4 + 1;
        assertEquals(new Complex(linearIndex, -linearIndex), slice.get(2, -1));
    }

    @Test
    void testSetLinearIndexingGetCartesianIndexing() {
        // linearIndex equal to cartesian index [1,3,2] in original array and [2,2] in slice:
        int arrayLinearIndex = (2 * 5 + (1 + 2)) * 4 + 1;
        int viewLinearIndex = 2 * 3 + 2;
        assertEquals(new Complex(arrayLinearIndex, -arrayLinearIndex), slice.get(2, -1));
        slice.set(new Complex(1, 1), viewLinearIndex);
        assertEquals(new Complex(1, 1), slice.get(2, -1));
        assertEquals(new Complex(1, 1), array.get(1, 3, 2));
    }

    @Test
    void testSetCartesianIndexingGetLinearIndexing() {
        // linearIndex equal to cartesian index [1,3,2] in original array and [2,2] in slice:
        int viewLinearIndex = 2 * 3 + 2;
        array.set(new Complex(1, 1), 1, -2, -1);
        assertEquals(new Complex(1, 1), slice.get(viewLinearIndex));
    }

    @Test
    void testWrongGetLinearIndexing() {
        Exception exception = assertThrows(ArrayIndexOutOfBoundsException.class, () -> slice.get(10));
        assertEquals(
            String.format(Errors.LINEAR_BOUNDS_ERROR, slice.length(), 10),
            exception.getMessage());
    }

    @Test
    void testWrongGetNegativeLinearIndexing() {
        Exception exception = assertThrows(ArrayIndexOutOfBoundsException.class, () -> slice.get(-11));
        assertEquals(
            String.format(Errors.LINEAR_BOUNDS_ERROR, slice.length(), -11),
            exception.getMessage());
    }

    @Test
    void testWrongGetCartesianIndexing() {
        Exception exception = assertThrows(ArrayIndexOutOfBoundsException.class, () -> slice.get(1,3));
        assertEquals(
            String.format(Errors.CARTESIAN_BOUNDS_ERROR, "3 × 3", "[1, 3]"),
            exception.getMessage());
    }

    @Test
    void testWrongGetNegativeCartesianIndexing() {
        Exception exception = assertThrows(ArrayIndexOutOfBoundsException.class, () -> slice.get(-4,1));
        assertEquals(
            String.format(Errors.CARTESIAN_BOUNDS_ERROR, "3 × 3", "[-4, 1]"),
            exception.getMessage());
    }

    @Test
    void testWrongSetLinearIndexing() {
        Complex zero = new Complex(0,0);
        Exception exception = assertThrows(ArrayIndexOutOfBoundsException.class,
            () -> slice.set(zero, 10));
        assertEquals(
            String.format(Errors.LINEAR_BOUNDS_ERROR, slice.length(), 10),
            exception.getMessage());
    }

    @Test
    void testWrongSetNegativeLinearIndexing() {
        Complex zero = new Complex(0,0);
        Exception exception = assertThrows(ArrayIndexOutOfBoundsException.class,
            () -> slice.set(zero, -11));
        assertEquals(
            String.format(Errors.LINEAR_BOUNDS_ERROR, slice.length(), -11),
            exception.getMessage());
    }

    @Test
    void testWrongSetCartesianIndexing() {
        Complex zero = new Complex(0,0);
        Exception exception = assertThrows(ArrayIndexOutOfBoundsException.class,
            () -> slice.set(zero, 1,3));
        assertEquals(
            String.format(Errors.CARTESIAN_BOUNDS_ERROR, "3 × 3", "[1, 3]"),
            exception.getMessage());
    }

    @Test
    void testWrongSetNegativeCartesianIndexing() {
        Complex zero = new Complex(0,0);
        Exception exception = assertThrows(ArrayIndexOutOfBoundsException.class,
            () -> slice.set(zero, -4,1));
        assertEquals(
            String.format(Errors.CARTESIAN_BOUNDS_ERROR, "3 × 3", "[-4, 1]"),
            exception.getMessage());
    }

    @Test
    void testGetDimensionMismatchTooMany() {
        Exception exception = assertThrows(IllegalArgumentException.class, () -> slice.get(1,1,0));
        assertEquals(
            String.format(Errors.DIMENSION_MISMATCH, 3, 2),
            exception.getMessage());
    }

    @Test
    void testGetDimensionMismatchNotEnough() {
        BartNDArray slice2 = array.slice("1:4", "1:4", ":");
        Exception exception = assertThrows(IllegalArgumentException.class, () -> slice2.get(1,1));
        assertEquals(
            String.format(Errors.DIMENSION_MISMATCH, 2, 3),
            exception.getMessage());
    }

    @Test
    void testSetDimensionMismatchTooMany() {
        Complex zero = new Complex(0,0);
        Exception exception = assertThrows(IllegalArgumentException.class,
            () -> slice.set(zero, 1,1,0));
        assertEquals(
            String.format(Errors.DIMENSION_MISMATCH, 3, 2),
            exception.getMessage());
    }

    @Test
    void testSetDimensionMismatchNotEnough() {
        Complex zero = new Complex(0,0);
        BartNDArray slice2 = array.slice("1:4", "1:4", ":");
        Exception exception = assertThrows(IllegalArgumentException.class,
            () -> slice2.set(zero, 1,1));
        assertEquals(
            String.format(Errors.DIMENSION_MISMATCH, 2, 3),
            exception.getMessage());
    }

    @Test
    void testEltype() {
        assertEquals(Complex.class, slice.dtype());
    }

    @Test
    void testToArray() {
        Complex[][] arr = (Complex[][])slice.toArray();
        for (int i = 0; i < slice.shape(0); i++)
            for (int j = 0; j < slice.shape(1); j++)
                assertEquals(array.get(1, 1 + i, j), arr[i][j]);
    }

    @Test
    void testEqual() {
        BartNDArray array2 = new BartComplexFloatNDArray(slice);
        assertEquals(slice, array2);
        array2.set(new Complex(0,0), 5);
        assertNotEquals(slice, array2);
    }

    @Test
    void testBasicNDArrayEqual() {
        ComplexNDArray<Float> array2 = new BasicComplexFloatNDArray(slice);
        assertEquals(slice, array2);
        array2.set(new Complex(0,0), 5);
        assertNotEquals(slice, array2);
    }

    @Test
    void testSizeNotEqual() {
        int[] shape = Arrays.copyOf(slice.shape(), slice.ndim() + 1);
        shape[slice.ndim()] = 1;
        ComplexNDArray<Float> array2 = new BasicComplexFloatNDArray(shape);
        array2.slice(":", ":", 0).copyFrom(slice);
        assertNotEquals(slice, array2);
    }

    @Test
    void testBartDimsEqual() {
        array.setBartDims(BartDimsEnum._00_READ, BartDimsEnum._01_PHS1, BartDimsEnum._02_PHS2);
        ComplexNDArray<Float> array2 = new BartComplexFloatNDArray(slice);
        assertEquals(slice, array2);
    }

    @Test
    void testBartDimsNotEqual() {
        ComplexNDArray<Float> array2 = new BasicComplexFloatNDArray(slice);
        array.setBartDims(BartDimsEnum._00_READ, BartDimsEnum._01_PHS1, BartDimsEnum._02_PHS2);
        assertNotEquals(slice, array2);
    }

    @Test
    void testBartDimsNotEqual2() {
        BartNDArray array2 = new BartComplexFloatNDArray(slice);
        array.setBartDims(BartDimsEnum._00_READ, BartDimsEnum._01_PHS1, BartDimsEnum._02_PHS2);
        array2.setBartDims(BartDimsEnum._00_READ, BartDimsEnum._02_PHS2);
        assertNotEquals(slice, array2);
    }

    @Test
    void testBartDimsNotEqual3() {
        ComplexNDArray<Float> array2 = new BartComplexFloatNDArray(slice);
        array.setBartDims(BartDimsEnum._00_READ, BartDimsEnum._01_PHS1, BartDimsEnum._02_PHS2);
        assertNotEquals(slice, array2);
    }

    @Test
    void testBartDimsNotEqual4() {
        BartComplexFloatNDArray array2 = new BartComplexFloatNDArray(slice);
        array2.setBartDims(BartDimsEnum._00_READ, BartDimsEnum._01_PHS1);
        assertNotEquals(slice, array2);
    }

    @Test
    void testHashCode() {
        assertThrows(UnsupportedOperationException.class, () -> { slice.hashCode(); });
    }

    @Test
    void testIterator() {
        int linearIndex = 0;
        for (Complex value : slice)
            assertEquals(slice.get(linearIndex++), value);
    }

    @Test
    void testStream() {
        final int[] linearIndex = new int[1];
        slice.stream().forEach((value) -> {
            assertEquals(slice.get(linearIndex[0]++), value);
        });
    }

    @Test
    void testParallelStream() {
        Complex sum = slice.stream().parallel()
            .reduce(new Complex(0,0), (acc, item) -> acc.add(item));
        Complex acc = new Complex(0,0);
        for (int i = 1; i < array.shape(1) - 1; i++)
            for (int j = 0; j < array.shape(2); j++)
                acc = acc.add(array.get(1, i, j));
        assertEquals(acc, sum);
    }

    @Test
    void testCollector() {
        final Complex one = new Complex(1,-1);
        NDArray<Complex> increased = slice.stream()
            .map((value) -> value.add(one))
            .collect(BartComplexFloatNDArray.getCollector(slice.shape()));
        for (int i = 0; i < slice.length(); i++)
            assertEquals(slice.get(i).add(one), increased.get(i));
    }

    @Test
    void testParallelCollector() {
        final Complex one = new Complex(1,-1);
        NDArray<?> increased = array.stream().parallel()
            .map((value) -> value.add(one))
            .collect(BartComplexFloatNDArray.getCollector(array.shape()));
        for (int i = 0; i < array.length(); i++)
            assertEquals(array.get(i).add(one), increased.get(i));
    }

    @Test
    void testToString() {
        String str = slice.toString();
        assertEquals(name() + " NDArrayView<Complex Float>(3 × 3)", str);
    }

    @Test
    void testcontentToString() {
        String str = slice.contentToString();
        String lineFormat = "%8.5e%+8.5ei\t%8.5e%+8.5ei\t%8.5e%+8.5ei\t%n";
        String expected = new StringBuilder()
            .append(name() + " NDArrayView<Complex Float>(3 × 3)" + System.lineSeparator())
            .append(String.format(lineFormat, 5.0e+00, -5.0e+00, 2.5e+01, -2.5e+01, 4.5e+01, -4.5e+01))
            .append(String.format(lineFormat, 9.0e+00, -9.0e+00, 2.9e+01, -2.9e+01, 4.9e+01, -4.9e+01))
            .append(String.format(lineFormat, 1.3e+01, -1.3e+01, 3.3e+01, -3.3e+01, 5.3e+01, -5.3e+01))
            .toString();
        assertEquals(expected, str);
    }

    @Test
    void testApply() {
        NDArray<Complex> slice2 = new BartComplexFloatNDArray(array).slice(1, "1:4", ":").apply(value -> value.atan());
        for (int i = 1; i < slice.length(); i++) {
            assertTrue(Math.abs(slice.get(i).atan().getReal() - slice2.get(i).getReal()) / slice2.get(i).getReal() < 1e-6);
            assertTrue(Math.abs(slice.get(i).atan().getImaginary() - slice2.get(i).getImaginary()) / slice2.get(i).getImaginary() < 1e-6);
        }
    }

    @Test
    void testApplyWithLinearIndices() {
        NDArray<Complex> slice2 = new BartComplexFloatNDArray(array).slice(1, "1:4", ":").applyWithLinearIndices((value, index) -> value.atan().add(index));
        for (int i = 1; i < slice.length(); i++) {
            assertTrue(Math.abs(slice.get(i).atan().add(i).getReal() - slice2.get(i).getReal()) / slice2.get(i).getReal() < 1e-6);
            assertTrue(Math.abs(slice.get(i).atan().add(i).getImaginary() - slice2.get(i).getImaginary()) / slice2.get(i).getImaginary() < 1e-6);
        }
    }

    @Test
    void testApplyWithCartesianIndex() {
        NDArray<Complex> slice2 = new BartComplexFloatNDArray(array).slice(1, "1:4", ":").applyWithCartesianIndices((value, indices) -> value.atan().add(indices[0]));
        for (int i = 0; i < slice.shape(0); i++)
            for (int j = 0; j < slice.shape(1); j++) {
                if (i == 0 && j == 0) continue;
                assertTrue(Math.abs(slice.get(i,j).atan().add(i).getReal() - slice2.get(i,j).getReal()) / slice2.get(i,j).getReal() < 1e-6);
                assertTrue(Math.abs(slice.get(i,j).atan().add(i).getImaginary() - slice2.get(i,j).getImaginary()) / slice2.get(i,j).getImaginary() < 1e-6);
            }
    }

    @Test
    void testMap() {
        NDArray<Complex> slice2 = slice.map(value -> value.atan());
        for (int i = 1; i < slice.length(); i++) {
            assertTrue(Math.abs(slice.get(i).atan().getReal() - slice2.get(i).getReal()) / slice2.get(i).getReal() < 1e-6);
            assertTrue(Math.abs(slice.get(i).atan().getImaginary() - slice2.get(i).getImaginary()) / slice2.get(i).getImaginary() < 1e-6);
        }
    }

    @Test
    void testMapWithLinearIndices() {
        NDArray<Complex> slice2 = slice.mapWithLinearIndices((value, index) -> value.atan().add(index));
        for (int i = 1; i < slice.length(); i++) {
            assertTrue(Math.abs(slice.get(i).atan().add(i).getReal() - slice2.get(i).getReal()) / slice2.get(i).getReal() < 1e-6);
            assertTrue(Math.abs(slice.get(i).atan().add(i).getImaginary() - slice2.get(i).getImaginary()) / slice2.get(i).getImaginary() < 1e-6);
        }
    }

    @Test
    void testMapWithCartesianIndex() {
        NDArray<Complex> slice2 = slice.mapWithCartesianIndices((value, indices) -> value.atan().add(indices[0]));
        for (int i = 0; i < slice.shape(0); i++)
            for (int j = 0; j < slice.shape(1); j++) {
                if (i == 0 && j == 0) continue;
                assertTrue(Math.abs(slice.get(i,j).atan().add(i).getReal() - slice2.get(i,j).getReal()) / slice2.get(i,j).getReal() < 1e-6);
                assertTrue(Math.abs(slice.get(i,j).atan().add(i).getImaginary() - slice2.get(i,j).getImaginary()) / slice2.get(i,j).getImaginary() < 1e-6);
            }
    }

    @Test
    void testForEachSequential() {
        AtomicInteger i = new AtomicInteger(0);
        slice.forEachSequential(value -> assertEquals(slice.get(i.getAndIncrement()), value));
    }

    @Test
    void testForEachWithLinearIndices() {
        slice.forEachWithLinearIndices((value, index) -> assertEquals(slice.get(index), value));
    }

    @Test
    void testForEachWithCartesianIndex() {
        slice.forEachWithCartesianIndices((value, indices) -> assertEquals(slice.get(indices), value));
    }

    @Test
    void testAdd() {
        BartNDArray array2 = new BartComplexFloatNDArray(slice);
        BartNDArray array3 = slice.add(array2);
        for (int i = 0; i < slice.length(); i++)
            assertEquals(slice.get(i).multiply(2), array3.get(i));
    }

    @Test
    void testAddArrayToSlice() {
        BartNDArray array2 = new BartComplexFloatNDArray(slice);
        BartNDArray array3 = slice.add(array2);
        for (int i = 0; i < slice.length(); i++)
            assertEquals(slice.get(i).multiply(2), array3.get(i));
    }

    @Test
    void testAddSliceToArray() {
        BartNDArray array2 = new BartComplexFloatNDArray(slice);
        BartNDArray array3 = array2.add(slice);
        for (int i = 0; i < slice.length(); i++)
            assertEquals(slice.get(i).multiply(2), array3.get(i));
    }

    @Test
    void testAddSliceToSlice() {
        BartNDArray slice2 = array.slice(1, "1:4", ":");
        BartNDArray array3 = slice2.add(slice);
        for (int i = 0; i < slice.length(); i++)
            assertEquals(slice.get(i).multiply(2), array3.get(i));
    }

    @Test
    void testAddScalar() {
        BartNDArray slice2 = slice.add(5);
        for (int i = 0; i < slice.length(); i++)
            assertEquals(slice.get(i).add(5), slice2.get(i));
    }

    @Test
    void testAddMultiple() {
        BartNDArray array2 = new BartComplexFloatNDArray(array);
        BartNDArray slice2 = array2.slice(1, "1:4", ":");
        BartNDArray array3 = slice2.add(slice, 5.3, slice2, new Complex(3,1));
        for (int i = 0; i < slice.length(); i++) {
            Complex expected = slice.get(i).multiply(3).add(new Complex(5.3 + 3,1));
            assertTrue(expected.subtract(array3.get(i)).abs() < 1e5);
        }
    }

    @Test
    void testAddInplace() {
        BartNDArray array2 = new BartComplexFloatNDArray(array);
        BartNDArray slice2 = array2.slice(1, "1:4", ":");
        slice2.addInplace(slice);
        for (int i = 0; i < slice.length(); i++)
            assertEquals(slice.get(i).multiply(2), slice2.get(i));
    }

    @Test
    void testAddInplaceScalar() {
        BartNDArray array2 = new BartComplexFloatNDArray(array);
        BartNDArray slice2 = array2.slice(1, "1:4", ":");
        slice2.addInplace(5);
        for (int i = 0; i < slice.length(); i++)
            assertEquals(slice.get(i).add(5), slice2.get(i));
    }

    @Test
    void testAddInplaceMultiple() {
        BartNDArray array2 = new BartComplexFloatNDArray(array);
        BartNDArray slice2 = array2.slice(1, "1:4", ":");
        slice2.addInplace(slice, 5.3, slice2, new Complex(3,1));
        for (int i = 0; i < slice.length(); i++) {
            Complex expected = slice.get(i).multiply(3).add(new Complex(5.3 + 3,1));
            assertTrue(expected.subtract(slice2.get(i)).abs() < 1e5);
        }
    }

    @Test
    void test0Norm() {
        slice.slice(":", 0).fill(0);
        double norm = slice.stream()
            .filter(value -> value != Complex.ZERO)
            .count();
        assertEquals(norm, slice.norm(0));
    }

    @Test
    void test1Norm() {
        double norm = (float)slice.stream()
            .map(value -> (float)value.abs())
            .reduce((float)0., (acc, item) -> acc + item);
        assertTrue(Math.abs(norm - slice.norm(1)) / norm < 1e-6);
    }

    @Test
    void test2Norm() {
        double norm = (float)Math.sqrt(slice.stream()
            .map(value -> (float)Math.pow(value.abs(), 2))
            .reduce((float)0., (acc, item) -> acc + item));
        assertTrue(Math.abs(norm - slice.norm()) / norm < 1e-6);
    }

    @Test
    void testPQuasinorm() {
        double norm = (float)Math.pow(slice.stream()
            .map(value -> (float)Math.pow(value.abs(), 0.5))
            .reduce((float)0., (acc, item) -> acc + item), 2);
        assertTrue(Math.abs(norm - slice.norm(0.5)) / norm < 5e-6);
    }

    @Test
    void testPNorm() {
        double norm = (float)Math.pow(slice.stream()
            .map(value -> (float)Math.pow(value.abs(), 3.5))
            .reduce((float)0., (acc, item) -> acc + item), 1 / 3.5);
        assertTrue(Math.abs(norm - slice.norm(3.5)) / norm < 5e-6);
    }

    @Test
    void testInfNorm() {
        double norm = (float)slice.stream()
            .mapToDouble(value -> value.abs())
            .max().getAsDouble();
        assertEquals(norm, slice.norm(Double.POSITIVE_INFINITY));
    }

    @Test
    void testCopy() {
        BartNDArray array2 = slice.copy();
        for (int i = 0; i < slice.length(); i++)
            assertEquals(slice.get(i), array2.get(i));
        array2.set(new Complex(0,0), 5);
        assertNotEquals(slice.get(5), array2.get(5));
    }

    @Test
    void testSimilar() {
        BartNDArray array2 = slice.similar();
        assertArrayEquals(slice.shape(), array2.shape());
        array2.forEach(value-> assertEquals(Complex.ZERO, value));
        array2.set(new Complex(0,0), 5);
        assertNotEquals(slice.get(5), array2.get(5));
    }

    @Test
    void testFillComplex() {
        slice.fill(new Complex(3,3));
        for (Complex elem : slice)
            assertEquals(new Complex(3, 3), elem);
        BartNDArray slice2 = array.slice(0, ":", ":");
        for (Complex elem : slice2)
            assertNotEquals(new Complex(3, 3), elem);
    }

    @Test
    void testFillReal() {
        slice.fill(3);
        for (Complex elem : slice)
            assertEquals(new Complex(3, 0), elem);
        BartNDArray slice2 = array.slice(0, ":", ":");
        for (Complex elem : slice2)
            assertNotEquals(new Complex(3, 0), elem);
    }

    @Test
    void testPermuteDimsAndToArray() {
        BartNDArray pArray = slice.permuteDims(1,0);
        assertTrue(pArray.copy() instanceof BartComplexFloatNDArray);
        Complex[][] arr = (Complex[][])pArray.toArray();
        for (int i = 0; i < pArray.shape(0); i++)
            for (int j = 0; j < pArray.shape(1); j++)
                assertEquals(array.get(1, 1 + i, j), arr[j][i]);
    }

    @Test
    void testMaskSlice() {
        NDArray<Byte> mask = new BasicByteNDArray(slice.abs().map(value -> value > 20 ? (float)1 : (float)0));
        BartNDArray masked = slice.mask(mask);
        masked.forEachSequential((value) -> assertTrue(value.abs() > 20));
        masked.fill(0);
        slice.forEachSequential(value -> assertTrue(value.abs() <= 20));
        assertTrue(masked.copy() instanceof BartComplexFloatNDArray);
    }

    @Test
    void testMaskSliceWithPredicate() {
        BartNDArray masked = slice.mask(value -> value.abs() > 20);
        masked.forEachSequential((value) -> assertTrue(value.abs() > 20));
        masked.fill(0);
        slice.forEachSequential(value -> assertTrue(value.abs() <= 20));
    }

    @Test
    void testMaskSliceWithPredicateWithLinearIndices() {
        BartNDArray masked = slice.maskWithLinearIndices((value, i) -> value.abs() > 20 && i < 10);
        masked.forEachWithLinearIndices((value, i) -> assertTrue(value.abs() > 20 && i < 10));
        masked.fill(0);
        slice.forEachWithLinearIndices((value, i) -> assertTrue(value.abs() <= 20 || i >= 10));
    }

    @Test
    void testMaskSliceWithPredicateWithCartesianIndices() {
        BartNDArray masked = slice.maskWithCartesianIndices((value, idx) -> value.abs() > 20 && idx[0] == 0);
        masked.forEachSequential(value -> assertTrue(value.abs() > 20));
        masked.fill(0);
        slice.forEachWithCartesianIndices((value, idx) -> assertTrue(value.abs() <= 20 || idx[0] != 0));
    }

    @Test
    void testConcatenate() {
        BartNDArray array2 = new BartComplexFloatNDArray(new int[]{5, 3}).fill(1);
        BartNDArray array3 = slice.concatenate(0, array2);
        for (int i = 0; i < slice.shape(0); i++)
            for (int j = 0; j < slice.shape(1); j++)
                assertEquals(slice.get(i, j), array3.get(i, j));
        for (int i = slice.shape(0); i < slice.shape(0) + array2.shape(0); i++)
            for (int j = 0; j < array2.shape(1); j++)
                assertEquals(new Complex(1, 0), array3.get(i, j));
    }

    @Test
    void testConcatenateMultiple() {
        BartNDArray array2 = slice.copy().fill(1).slice("1:1", ":");
        BartNDArray array3 = new BartComplexFloatNDArray(new int[]{3, 2}).permuteDims(1, 0);
        BartNDArray array4 = new BartComplexFloatNDArray(new int[]{9}).fill(new Complex(2, -2)).reshape(3, 3);
        BartNDArray array5 = slice.concatenate(0, array2, array3, array4);
        int start = 0;
        int end = slice.shape(0);
        for (int i = start; i < end; i++)
            for (int j = 0; j < slice.shape(1); j++)
                assertEquals(slice.get(i, j), array5.get(i, j));
        start = end;
        end += array2.shape(0);
        for (int i = start; i < end; i++)
            for (int j = 0; j < array2.shape(1); j++)
                assertEquals(new Complex(1, 0), array5.get(i, j));
        start = end;
        end += array3.shape(0);
        for (int i = start; i < end; i++)
            for (int j = 0; j < array2.shape(1); j++)
                assertEquals(new Complex(0, 0), array5.get(i, j));
        start = end;
        end += array4.shape(0);
        for (int i = start; i < end; i++)
            for (int j = 0; j < array2.shape(1); j++)
                assertEquals(new Complex(2, -2), array5.get(i, j));
    }

    @Test
    void testReal() {
        NDArray<Float> real = slice.real();
        slice.streamLinearIndices()
            .forEach(i -> assertEquals(slice.get(i).getReal(), real.get(i).doubleValue()));
    }

    @Test
    void testImag() {
        NDArray<Float> imag = slice.imaginary();
        slice.streamLinearIndices()
            .forEach(i -> assertEquals(slice.get(i).getImaginary(), imag.get(i).doubleValue()));
    }

    @Test
    void testAbs() {
        NDArray<Float> abs = slice.abs();
        slice.streamLinearIndices()
            .forEach(i -> assertTrue(slice.get(i).abs() - abs.get(i) < 1e-5));
    }

    @Test
    void testAngle() {
        NDArray<Float> argument = slice.argument();
        slice.streamLinearIndices()
            .forEach(i -> assertTrue(slice.get(i).getArgument() - argument.get(i) < 1e-5));
    }
}
