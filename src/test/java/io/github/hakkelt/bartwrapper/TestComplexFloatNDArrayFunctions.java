package io.github.hakkelt.bartwrapper;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.Arrays;

import org.apache.commons.math3.complex.Complex;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import io.github.hakkelt.ndarrays.ComplexNDArray;
import io.github.hakkelt.ndarrays.NDArray;
import io.github.hakkelt.ndarrays.basic.BasicComplexFloatNDArray;
import io.github.hakkelt.ndarrays.internal.Errors;

class TestComplexFloatNDArrayFunctions implements NameTrait {
    BartNDArray array;

    @BeforeEach
    void setup() {
        array = new BartComplexFloatNDArray(new int[]{ 4, 5, 3 });
        array.fillUsingLinearIndices(index -> new Complex(index, -index));
    }

    @Test
    void testGetNegativeLinearIndexing() {
        assertEquals(new Complex(55, -55), array.get(-5));
    }

    @Test
    void testGetNegativeCartesianIndexing() {
        int linearIndex = (2 * 5 + 2) * 4 + 2; // equal to cartesian index [2,2,2]
        assertEquals(new Complex(linearIndex, -linearIndex), array.get(2, -3, -1));
    }

    @Test
    void testSetLinearIndexingGetCartesianIndexing() {
        int linearIndex = (2 * 5 + 2) * 4 + 2; // equal to cartesian index [2,2,2]
        assertEquals(new Complex(linearIndex, -linearIndex), array.get(2, -3, -1));
        array.set(new Complex(1, 1), linearIndex);
        assertEquals(new Complex(1, 1), array.get(2, -3, -1));
    }

    @Test
    void testSetCartesianIndexingGetLinearIndexing() {
        int linearIndex = (2 * 5 + 2) * 4 + 2; // equal to cartesian index [2,2,2]
        array.set(new Complex(1, 1), 2, -3, -1);
        assertEquals(new Complex(1, 1), array.get(linearIndex));
    }

    @Test
    void testWrongGetLinearIndexing() {
        Exception exception = assertThrows(ArrayIndexOutOfBoundsException.class, () -> array.get(60));
        assertEquals(
            String.format(Errors.LINEAR_BOUNDS_ERROR, array.length(), 60),
            exception.getMessage());
    }

    @Test
    void testWrongGetNegativeLinearIndexing() {
        Exception exception = assertThrows(ArrayIndexOutOfBoundsException.class, () -> array.get(-61));
        assertEquals(
            String.format(Errors.LINEAR_BOUNDS_ERROR, array.length(), -61),
            exception.getMessage());
    }

    @Test
    void testWrongGetCartesianIndexing() {
        Exception exception = assertThrows(ArrayIndexOutOfBoundsException.class, () -> array.get(1,1,3));
        assertEquals(
            String.format(Errors.CARTESIAN_BOUNDS_ERROR, "4 × 5 × 3", "[1, 1, 3]"),
            exception.getMessage());
    }

    @Test
    void testWrongGetNegativeCartesianIndexing() {
        Exception exception = assertThrows(ArrayIndexOutOfBoundsException.class, () -> array.get(1,-6,1));
        assertEquals(
            String.format(Errors.CARTESIAN_BOUNDS_ERROR, "4 × 5 × 3", "[1, -6, 1]"),
            exception.getMessage());
    }

    @Test
    void testWrongSetLinearIndexing() {
        Complex zero = new Complex(0,0);
        Exception exception = assertThrows(ArrayIndexOutOfBoundsException.class,
            () -> array.set(zero, 60));
        assertEquals(
            String.format(Errors.LINEAR_BOUNDS_ERROR, array.length(), 60),
            exception.getMessage());
    }

    @Test
    void testWrongSetNegativeLinearIndexing() {
        Complex zero = new Complex(0,0);
        Exception exception = assertThrows(ArrayIndexOutOfBoundsException.class,
            () -> array.set(zero, -61));
        assertEquals(
            String.format(Errors.LINEAR_BOUNDS_ERROR, array.length(), -61),
            exception.getMessage());
    }

    @Test
    void testWrongSetCartesianIndexing() {
        Complex zero = new Complex(0,0);
        Exception exception = assertThrows(ArrayIndexOutOfBoundsException.class,
            () -> array.set(zero, 1,1,3));
        assertEquals(
            String.format(Errors.CARTESIAN_BOUNDS_ERROR, "4 × 5 × 3", "[1, 1, 3]"),
            exception.getMessage());
    }

    @Test
    void testWrongSetNegativeCartesianIndexing() {
        Complex zero = new Complex(0,0);
        Exception exception = assertThrows(ArrayIndexOutOfBoundsException.class,
            () -> array.set(zero, 1,-6,1));
        assertEquals(
            String.format(Errors.CARTESIAN_BOUNDS_ERROR, "4 × 5 × 3", "[1, -6, 1]"),
            exception.getMessage());
    }

    @Test
    void testGetDimensionMismatchTooMany() {
        Exception exception = assertThrows(IllegalArgumentException.class, () -> array.get(1,1,1,0));
        assertEquals(
            String.format(Errors.DIMENSION_MISMATCH, 4, 3),
            exception.getMessage());
    }

    @Test
    void testGetDimensionMismatchNotEnough() {
        Exception exception = assertThrows(IllegalArgumentException.class, () -> array.get(1,1));
        assertEquals(
            String.format(Errors.DIMENSION_MISMATCH, 2, 3),
            exception.getMessage());
    }

    @Test
    void testSetDimensionMismatchTooMany() {
        Complex zero = new Complex(0,0);
        Exception exception = assertThrows(IllegalArgumentException.class,
            () -> array.set(zero, 1,1,1,0));
        assertEquals(
            String.format(Errors.DIMENSION_MISMATCH, 4, 3),
            exception.getMessage());
    }

    @Test
    void testSetDimensionMismatchNotEnough() {
        Complex zero = new Complex(0,0);
        Exception exception = assertThrows(IllegalArgumentException.class,
            () -> array.set(zero, 1,1));
        assertEquals(
            String.format(Errors.DIMENSION_MISMATCH, 2, 3),
            exception.getMessage());
    }

    @Test
    void testEltype() {
        assertEquals(Complex.class, array.dtype());
    }

    @Test
    void testToArray() {
        Complex[][][] arr = (Complex[][][])array.toArray();
        int linearIndex = 0;
        for (int i = 0; i < arr[0][0].length; i++)
            for (int j = 0; j < arr[0].length; j++)
                for (int k = 0; k < arr.length; k++) {
                    assertEquals(new Complex(linearIndex, -linearIndex), arr[k][j][i]);
                    linearIndex++;
                }
    }

    @Test
    void testEqual() {
        BartNDArray array2 = new BartComplexFloatNDArray(array);
        assertEquals(array, array2);
        array2.set(new Complex(0,0), 10);
        assertNotEquals(array, array2);
    }

    @Test
    void testBasicNDArrayEqual() {
        ComplexNDArray<Float> array2 = new BasicComplexFloatNDArray(array);
        assertEquals(array, array2);
        array2.set(new Complex(0,0), 10);
        assertNotEquals(array, array2);
    }

    @Test
    void testSizeNotEqual() {
        int[] shape = Arrays.copyOf(array.shape(), array.ndim() + 1);
        shape[array.ndim()] = 1;
        ComplexNDArray<Float> array2 = new BasicComplexFloatNDArray(shape);
        array2.slice(":", ":", ":", 0).copyFrom(array);
        assertNotEquals(array, array2);
    }

    @Test
    void testBartDimsNotEqual() {
        ComplexNDArray<Float> array2 = new BasicComplexFloatNDArray(array);
        array.setBartDims(BartDimsEnum._00_READ, BartDimsEnum._01_PHS1, BartDimsEnum._02_PHS2);
        assertNotEquals(array, array2);
    }

    @Test
    void testBartDimsNotEqual2() {
        array.setBartDims(BartDimsEnum._00_READ, BartDimsEnum._01_PHS1, BartDimsEnum._02_PHS2);
        ComplexNDArray<Float> array2 = new BartComplexFloatNDArray(array);
        assertEquals(array, array2);
    }

    @Test
    void testBartDimsNotEqual3() {
        BartComplexFloatNDArray array2 = new BartComplexFloatNDArray(array);
        array2.setBartDims(BartDimsEnum._00_READ, BartDimsEnum._01_PHS1, BartDimsEnum._02_PHS2);
        assertNotEquals(array, array2);
    }

    @Test
    void testBartDimsNotEqual4() {
        BartComplexFloatNDArray array2 = new BartComplexFloatNDArray(array);
        array.setBartDims(BartDimsEnum._00_READ, BartDimsEnum._03_COIL, BartDimsEnum._04_MAPS);
        array2.setBartDims(BartDimsEnum._00_READ, BartDimsEnum._01_PHS1, BartDimsEnum._02_PHS2);
        assertNotEquals(array, array2);
    }

    @Test
    void testHashCode() {
        assertThrows(UnsupportedOperationException.class, () -> { array.hashCode(); });
    }

    @Test
    void testIterator() {
        int linearIndex = 0;
        for (Complex value : array) {
            assertEquals(new Complex(linearIndex, -linearIndex), value);
            linearIndex++;
        }
    }

    @Test
    void testStream() {
        final int[] linearIndex = new int[1];
        array.stream().forEach((value) -> {
            assertEquals(new Complex(linearIndex[0], -linearIndex[0]), value);
            linearIndex[0]++;
        });
    }

    @Test
    void testParallelStream() {
        Complex sum = array.stream().parallel()
            .reduce(new Complex(0,0), (acc, item) -> acc.add(item));
        int GaussSum = (0 + array.length() - 1) * array.length() / 2;
        assertEquals(new Complex(GaussSum, -GaussSum), sum);
    }

    @Test
    void testCollector() {
        final Complex one = new Complex(1,-1);
        NDArray<Complex> increased = array.stream()
            .map((value) -> value.add(one))
            .collect(BartComplexFloatNDArray.getCollector(array.shape()));
        for (int i = 0; i < array.length(); i++)
            assertEquals(array.get(i).add(one), increased.get(i));
    }

    @Test
    void testParallelCollector() {
        final Complex one = new Complex(1,-1);
        NDArray<Complex> increased = array.stream().parallel()
            .map((value) -> value.add(one))
            .collect(BartComplexFloatNDArray.getCollector(array.shape()));
        for (int i = 0; i < array.length(); i++)
            assertEquals(array.get(i).add(one), increased.get(i));
    }

    @Test
    void testToString() {
        String str = array.toString();
        assertEquals(name() + " NDArray<Complex Float>(4 × 5 × 3)", str);
    }

    @Test
    void testcontentToString() {
        String str = array.contentToString();
        String lineFormat = "%8.3e%+8.3ei\t%8.3e%+8.3ei\t%8.3e%+8.3ei\t%8.3e%+8.3ei\t%8.3e%+8.3ei\t%n";
        String expected = new StringBuilder()
            .append(name() + " NDArray<Complex Float>(4 × 5 × 3)" + System.lineSeparator())
            .append("[:, :, 0] =" + System.lineSeparator())
            .append(String.format(lineFormat, 0.0e+00, +0.0e+00, 4.0e+00, -4.0e+00, 8.0e+00, -8.0e+00, 1.2e+01, -1.2e+01, 1.6e+01, -1.6e+01))
            .append(String.format(lineFormat, 1.0e+00, -1.0e+00, 5.0e+00, -5.0e+00, 9.0e+00, -9.0e+00, 1.3e+01, -1.3e+01, 1.7e+01, -1.7e+01))
            .append(String.format(lineFormat, 2.0e+00, -2.0e+00, 6.0e+00, -6.0e+00, 1.0e+01, -1.0e+01, 1.4e+01, -1.4e+01, 1.8e+01, -1.8e+01))
            .append(String.format(lineFormat, 3.0e+00, -3.0e+00, 7.0e+00, -7.0e+00, 1.1e+01, -1.1e+01, 1.5e+01, -1.5e+01, 1.9e+01, -1.9e+01))
            .append(System.lineSeparator())
            .append("[:, :, 1] =" + System.lineSeparator())
            .append(String.format(lineFormat, 2.0e+01, -2.0e+01, 2.4e+01, -2.4e+01, 2.8e+01, -2.8e+01, 3.2e+01, -3.2e+01, 3.6e+01, -3.6e+01))
            .append(String.format(lineFormat, 2.1e+01, -2.1e+01, 2.5e+01, -2.5e+01, 2.9e+01, -2.9e+01, 3.3e+01, -3.3e+01, 3.7e+01, -3.7e+01))
            .append(String.format(lineFormat, 2.2e+01, -2.2e+01, 2.6e+01, -2.6e+01, 3.0e+01, -3.0e+01, 3.4e+01, -3.4e+01, 3.8e+01, -3.8e+01))
            .append(String.format(lineFormat, 2.3e+01, -2.3e+01, 2.7e+01, -2.7e+01, 3.1e+01, -3.1e+01, 3.5e+01, -3.5e+01, 3.9e+01, -3.9e+01))
            .append(System.lineSeparator())
            .append("[:, :, 2] =" + System.lineSeparator())
            .append(String.format(lineFormat, 4.0e+01, -4.0e+01, 4.4e+01, -4.4e+01, 4.8e+01, -4.8e+01, 5.2e+01, -5.2e+01, 5.6e+01, -5.6e+01))
            .append(String.format(lineFormat, 4.1e+01, -4.1e+01, 4.5e+01, -4.5e+01, 4.9e+01, -4.9e+01, 5.3e+01, -5.3e+01, 5.7e+01, -5.7e+01))
            .append(String.format(lineFormat, 4.2e+01, -4.2e+01, 4.6e+01, -4.6e+01, 5.0e+01, -5.0e+01, 5.4e+01, -5.4e+01, 5.8e+01, -5.8e+01))
            .append(String.format(lineFormat, 4.3e+01, -4.3e+01, 4.7e+01, -4.7e+01, 5.1e+01, -5.1e+01, 5.5e+01, -5.5e+01, 5.9e+01, -5.9e+01))
            .append(System.lineSeparator())
            .toString();
        assertEquals(expected, str);
    }

    @Test
    void testApply() {
        BartNDArray array2 = new BartComplexFloatNDArray(array).apply(value -> value.atan());
        for (int i = 1; i < array.length(); i++) {
            assertTrue(Math.abs(array.get(i).atan().getReal() - array2.get(i).getReal()) / array2.get(i).getReal() < 1e-6);
            assertTrue(Math.abs(array.get(i).atan().getImaginary() - array2.get(i).getImaginary()) / array2.get(i).getImaginary() < 1e-6);
        }
    }

    @Test
    void testApplyWithLinearIndices() {
        BartNDArray array2 = new BartComplexFloatNDArray(array).applyWithLinearIndices((value, index) -> value.atan().add(index));
        for (int i = 1; i < array.length(); i++) {
            assertTrue(Math.abs(array.get(i).atan().add(i).getReal() - array2.get(i).getReal()) / array2.get(i).getReal() < 1e-6);
            assertTrue(Math.abs(array.get(i).atan().add(i).getImaginary() - array2.get(i).getImaginary()) / array2.get(i).getImaginary() < 1e-6);
        }
    }

    @Test
    void testApplyWithCartesianIndex() {
        BartNDArray array2 = new BartComplexFloatNDArray(array).applyWithCartesianIndices((value, indices) -> value.atan().add(indices[0]));
        for (int i = 0; i < array.shape(0); i++)
            for (int j = 0; j < array.shape(1); j++)
                for (int k = 0; k < array.shape(2); k++) {
                    if (i == 0 && j == 0 && k == 0) continue;
                    assertTrue(Math.abs(array.get(i,j,k).atan().add(i).getReal() - array2.get(i,j,k).getReal()) / array2.get(i,j,k).getReal() < 1e-6);
                    assertTrue(Math.abs(array.get(i,j,k).atan().add(i).getImaginary() - array2.get(i,j,k).getImaginary()) / array2.get(i,j,k).getImaginary() < 1e-6);
                }
    }

    @Test
    void testMap() {
        BartNDArray array2 = array.map(value -> value.atan());
        for (int i = 1; i < array.length(); i++) {
            assertTrue(Math.abs(array.get(i).atan().getReal() - array2.get(i).getReal()) / array2.get(i).getReal() < 1e-6);
            assertTrue(Math.abs(array.get(i).atan().getImaginary() - array2.get(i).getImaginary()) / array2.get(i).getImaginary() < 1e-6);
        }
    }

    @Test
    void testMapWithLinearIndices() {
        BartNDArray array2 = array.mapWithLinearIndices((value, index) -> value.atan().add(index));
        for (int i = 1; i < array.length(); i++) {
            assertTrue(Math.abs(array.get(i).atan().add(i).getReal() - array2.get(i).getReal()) / array2.get(i).getReal() < 1e-6);
            assertTrue(Math.abs(array.get(i).atan().add(i).getImaginary() - array2.get(i).getImaginary()) / array2.get(i).getImaginary() < 1e-6);
        }
    }

    @Test
    void testMapWithCartesianIndex() {
        BartNDArray array2 = array.mapWithCartesianIndices((value, indices) -> value.atan().add(indices[0]));
        for (int i = 0; i < array.shape(0); i++)
            for (int j = 0; j < array.shape(1); j++)
                for (int k = 0; k < array.shape(2); k++) {
                    if (i == 0 && j == 0 && k == 0) continue;
                    assertTrue(Math.abs(array.get(i,j,k).atan().add(i).getReal() - array2.get(i,j,k).getReal()) / array2.get(i,j,k).getReal() < 1e-6);
                    assertTrue(Math.abs(array.get(i,j,k).atan().add(i).getImaginary() - array2.get(i,j,k).getImaginary()) / array2.get(i,j,k).getImaginary() < 1e-6);
                }
    }

    @Test
    void testForEachWithLinearIndices() {
        array.forEachWithLinearIndices((value, index) -> assertEquals(array.get(index), value));
    }

    @Test
    void testForEachWithCartesianIndex() {
        array.forEachWithCartesianIndices((value, indices) -> assertEquals(array.get(indices), value));
    }

    @Test
    void testAdd() {
        BartNDArray array2 = new BartComplexFloatNDArray(array);
        BartNDArray array3 = array.add(array2);
        for (int i = 0; i < array.length(); i++)
            assertEquals(array.get(i).multiply(2), array3.get(i));
    }

    @Test
    void testAddScalar() {
        BartNDArray array2 = array.add(5);
        for (int i = 0; i < array.length(); i++)
            assertEquals(array.get(i).add(5), array2.get(i));
    }

    @Test
    void testAddMultiple() {
        BartNDArray array2 = new BartComplexFloatNDArray(array);
        BartNDArray array3 = array.add(array2, 5.3, array2, new Complex(3,1));
        for (int i = 0; i < array.length(); i++) {
            Complex expected = array.get(i).multiply(3).add(new Complex(5.3 + 3,1));
            assertTrue(expected.subtract(array3.get(i)).abs() < 1e5);
        }
    }

    @Test
    void testAddInplace() {
        BartNDArray array2 = new BartComplexFloatNDArray(array);
        array2.addInplace(array);
        for (int i = 0; i < array.length(); i++)
            assertEquals(array.get(i).multiply(2), array2.get(i));
    }

    @Test
    void testAddInplaceScalar() {
        BartNDArray array2 = new BartComplexFloatNDArray(array);
        array2.addInplace(5);
        for (int i = 0; i < array.length(); i++)
            assertEquals(array.get(i).add(5), array2.get(i));
    }

    @Test
    void testAddInplaceMultiple() {
        BartNDArray array2 = new BartComplexFloatNDArray(array);
        array2.addInplace(array, 5.3, array2, new Complex(3,1));
        for (int i = 0; i < array.length(); i++) {
            Complex expected = array.get(i).multiply(3).add(new Complex(5.3 + 3,1));
            assertTrue(expected.subtract(array2.get(i)).abs() < 1e5);
        }
    }

    @Test
    void testSubtract() {
        BartNDArray array2 = new BartComplexFloatNDArray(array);
        BartNDArray array3 = array.subtract(array2);
        for (int i = 0; i < array.length(); i++)
            assertEquals(new Complex(0,0), array3.get(i));
    }

    @Test
    void testSubtractScalar() {
        BartNDArray array2 = array.subtract(5);
        for (int i = 0; i < array.length(); i++)
            assertEquals(array.get(i).subtract(5), array2.get(i));
    }

    @Test
    void testSubtractMultiple() {
        BartNDArray array2 = new BartComplexFloatNDArray(array);
        BartNDArray array3 = array.subtract(array2, 5.3, array2, new Complex(3,1));
        for (int i = 0; i < array.length(); i++) {
            Complex expected = array.get(i).multiply(-1).subtract(new Complex(5.3 + 3,1));
            assertTrue(expected.subtract(array3.get(i)).abs() < 1e5);
        }
    }

    @Test
    void testSubtractInplace() {
        BartNDArray array2 = new BartComplexFloatNDArray(array);
        array2.subtractInplace(array);
        for (int i = 0; i < array.length(); i++)
            assertEquals(new Complex(0,0), array2.get(i));
    }

    @Test
    void testSubtractInplaceScalar() {
        BartNDArray array2 = new BartComplexFloatNDArray(array);
        array2.subtractInplace(5);
        for (int i = 0; i < array.length(); i++)
            assertEquals(array.get(i).subtract(5), array2.get(i));
    }

    @Test
    void testSubtractInplaceMultiple() {
        BartNDArray array2 = new BartComplexFloatNDArray(array);
        array2.subtractInplace(array, 5.3, array2, new Complex(3,1));
        for (int i = 0; i < array.length(); i++) {
            Complex expected = array.get(i).multiply(-1).subtract(new Complex(5.3 + 3,1));
            assertTrue(expected.subtract(array2.get(i)).abs() < 1e5);
        }
    }

    @Test
    void testMultiply() {
        BartNDArray array2 = new BartComplexFloatNDArray(array);
        BartNDArray array3 = array.multiply(array2);
        for (int i = 0; i < array.length(); i++)
            assertEquals(array.get(i).multiply(array.get(i)), array3.get(i));
    }

    @Test
    void testMultiplyScalar() {
        BartNDArray array2 = array.multiply(5);
        for (int i = 0; i < array.length(); i++)
            assertEquals(array.get(i).multiply(5), array2.get(i));
    }

    @Test
    void testMultiplyMultiple() {
        BartNDArray array2 = new BartComplexFloatNDArray(array);
        BartNDArray array3 = array.multiply(array, 5.3, array2, new Complex(3,1));
        for (int i = 0; i < array.length(); i++) {
            Complex expected = array.get(i).multiply(array.get(i)).multiply((float)5.3)
                .multiply(array2.get(i)).multiply(new Complex(3,1));
            assertTrue(expected.subtract(array3.get(i)).abs() < 1e7);
        }
    }

    @Test
    void testMultiplyInplace() {
        BartNDArray array2 = new BartComplexFloatNDArray(array);
        array2.multiplyInplace(array);
        for (int i = 0; i < array.length(); i++)
            assertEquals(array.get(i).multiply(array.get(i)), array2.get(i));
    }

    @Test
    void testMultiplyInplaceScalar() {
        BartNDArray array2 = new BartComplexFloatNDArray(array);
        array2.multiplyInplace(5);
        for (int i = 0; i < array.length(); i++)
            assertEquals(array.get(i).multiply(5), array2.get(i));
    }

    @Test
    void testMultiplyInplaceMultiple() {
        BartNDArray array2 = new BartComplexFloatNDArray(array);
        array2.multiplyInplace(array, 5.3, array2, new Complex(3,1));
        for (int i = 0; i < array.length(); i++) {
            Complex expected = array.get(i).multiply(array.get(i)).multiply((float)5.3)
                .multiply(array.get(i)).multiply(new Complex(3,1));
            if (expected.abs() == 0)
                assertTrue(expected.subtract(array2.get(i)).abs() < 1e7);
            else
                assertTrue(expected.subtract(array2.get(i)).divide(expected.abs()).abs() < 1e7);
        }
    }

    @Test
    void testDivide() {
        BartNDArray array2 = new BartComplexFloatNDArray(array);
        BartNDArray array3 = array.divide(array2);
        for (int i = 0; i < array.length(); i++) {
            if (array.get(i).equals(new Complex(0,0)))
                assertTrue(array3.get(i).isNaN());
            else
                assertEquals(new Complex(1,0), array3.get(i));
        }
    }

    @Test
    void testDivideScalar() {
        BartNDArray array2 = array.divide(5);
        for (int i = 0; i < array.length(); i++)
            assertTrue(array.get(i).divide(5).subtract(array2.get(i)).abs() < 1e-5);
    }

    @Test
    void testDivideMultiple() {
        BartNDArray array2 = new BartComplexFloatNDArray(array);
        BartNDArray array3 = array.divide(array, 5.3, array2, new Complex(3,1));
        for (int i = 0; i < array.length(); i++) {
            Complex expected = array.get(i).divide(array.get(i)).divide((float)5.3)
                .divide(array2.get(i)).divide(new Complex(3,1));
            if (expected.isNaN())
                assertTrue(array3.get(i).isNaN());
            else
                assertTrue(expected.subtract(array3.get(i)).abs() < 1e7);
        }
    }

    @Test
    void testDivideInplace() {
        BartNDArray array2 = new BartComplexFloatNDArray(array);
        array2.divideInplace(array);
        for (int i = 0; i < array.length(); i++)
        if (array.get(i).equals(new Complex(0,0)))
            assertTrue(array2.get(i).isNaN());
        else
            assertEquals(new Complex(1,0), array2.get(i));
    }

    @Test
    void testDivideInplaceScalar() {
        BartNDArray array2 = new BartComplexFloatNDArray(array);
        array2.divideInplace(5);
        for (int i = 0; i < array.length(); i++)
            assertTrue(array.get(i).divide(5).subtract(array2.get(i)).abs() < 1e-5);
    }

    @Test
    void testDivideInplaceMultiple() {
        BartNDArray array2 = new BartComplexFloatNDArray(array);
        array2.divideInplace(array, 5.3, array2, new Complex(3,1));
        for (int i = 0; i < array.length(); i++) {
            Complex expected = array.get(i).divide(array.get(i)).divide((float)5.3)
                .divide(array.get(i)).divide(new Complex(3,1));
            if (expected.isNaN())
                assertTrue(array2.get(i).isNaN());
            else
                assertTrue(expected.subtract(array2.get(i)).abs() < 1e7);
        }
    }

    @Test
    void testSum() {
        Complex sum = array.sum();
        int GaussSum = (0 + array.length() - 1) * array.length() / 2;
        assertEquals(new Complex(GaussSum, -GaussSum), sum);
    }

    @Test
    void testSum1D() {
        BartNDArray sum = array.sum(1);
        for (int i = 0; i < sum.shape(0); i++) {
            for (int j = 0; j < sum.shape(1); j++) {
                double GaussSum = (array.get(i,0,j).getReal() + array.get(i,-1,j).getReal()) * 5 / 2;
                assertEquals(new Complex(GaussSum, -GaussSum), sum.get(i,j));
            }
        }
    }

    @Test
    void testSum2D() {
        BartNDArray sum = array.sum(2, 1);
        for (int i = 0; i < sum.length(); i++) {
            double GaussSum = (array.get(i,0,0).getReal() + array.get(i,-1,-1).getReal()) * (5 * 3) / 2;
            assertEquals(new Complex(GaussSum, -GaussSum), sum.get(i));
        }
    }

    @Test
    void test0Norm() {
        array.slice(":", 0, ":").fill(0);
        double norm = array.stream()
            .filter(value -> !value.equals(Complex.ZERO))
            .count();
        assertEquals(norm, array.norm(0));
    }

    @Test
    void test1Norm() {
        double norm = (float)array.stream()
            .map(value -> (float)value.abs())
            .reduce((float)0., (acc, item) -> acc + item);
        assertTrue(Math.abs(norm - array.norm(1)) / norm < 1e-6);
    }

    @Test
    void test2Norm() {
        double norm = (float)Math.sqrt(array.stream()
            .map(value -> (float)Math.pow(value.abs(), 2))
            .reduce((float)0., (acc, item) -> acc + item));
        assertTrue(Math.abs(norm - array.norm()) / norm < 1e-6);
    }

    @Test
    void testPQuasinorm() {
        double norm = (float)Math.pow(array.stream()
            .map(value -> (float)Math.pow(value.abs(), 0.5))
            .reduce((float)0., (acc, item) -> acc + item), 2);
        assertTrue(Math.abs(norm - array.norm(0.5)) / norm < 5e-6);
    }

    @Test
    void testPNorm() {
        double norm = (float)Math.pow(array.stream()
            .map(value -> (float)Math.pow(value.abs(), 3.5))
            .reduce((float)0., (acc, item) -> acc + item), 1 / 3.5);
        assertTrue(Math.abs(norm - array.norm(3.5)) / norm < 1e-6);
    }

    @Test
    void testInfNorm() {
        double norm = (float)array.stream()
            .mapToDouble(value -> value.abs())
            .max().getAsDouble();
        assertEquals(norm, array.norm(Double.POSITIVE_INFINITY));
    }

    @Test
    void testCopy() {
        BartNDArray array2 = array.copy();
        for (int i = 0; i < array.length(); i++)
            assertEquals(array.get(i), array2.get(i));
        array2.set(new Complex(0,0), 5);
        assertNotEquals(array.get(5), array2.get(5));
    }

    @Test
    void testCopyFrom() {
        BartNDArray array2 = array.similar().copyFrom(array);
        for (int i = 0; i < array.length(); i++)
            assertEquals(array.get(i), array2.get(i));
        array2.set(new Complex(0,0), 5);
        assertNotEquals(array.get(5), array2.get(5));
    }

    @Test
    void testSimilar() {
        BartNDArray array2 = array.similar();
        assertArrayEquals(array.shape(), array2.shape());
        array2.forEach(value-> assertEquals(Complex.ZERO, value));
        array2.set(new Complex(0,0), 5);
        assertNotEquals(array.get(5), array2.get(5));
    }

    @Test
    void testFillComplex() {
        array.fill(new Complex(3,3));
        for (Complex elem : array)
            assertEquals(new Complex(3, 3), elem);
    }

    @Test
    void testFillReal() {
        array.fill(3);
        for (Complex elem : array)
            assertEquals(new Complex(3, 0), elem);
    }

    @Test
    void testFillFloat() {
        array.fill(Float.valueOf(3));
        for (Complex elem : array)
            assertEquals(new Complex(3, 0), elem);
    }

    @Test
    void testFillDouble() {
        array.fill(3.);
        for (Complex elem : array)
            assertEquals(new Complex(3, 0), elem);
    }

    @Test
    void testPermuteDimsTooShortPermutationVector() {
        Exception exception = assertThrows(IllegalArgumentException.class, () -> array.permuteDims(0,2));
        assertEquals(
            String.format(Errors.PERMUTATOR_SHAPE_MISMATCH, "[0, 2]", "4 × 5 × 3"),
            exception.getMessage());
    }

    @Test
    void testPermuteDimsTooLongPermutationVector() {
        Exception exception = assertThrows(IllegalArgumentException.class, () -> array.permuteDims(0,2,1,4));
        assertEquals(
            String.format(Errors.PERMUTATOR_SHAPE_MISMATCH, "[0, 2, 1, 4]", "4 × 5 × 3"),
            exception.getMessage());
    }

    @Test
    void testPermuteDimsRepeatedDimension() {
        Exception exception = assertThrows(IllegalArgumentException.class, () -> array.permuteDims(0,1,1));
        assertEquals(
            String.format(Errors.INVALID_PERMUTATOR, "[0, 1, 1]", "4 × 5 × 3"),
            exception.getMessage());
    }
    @Test
    void testConcatenate() {
        BartNDArray array2 = new BartComplexFloatNDArray(new int[]{4, 2, 3}).fill(1);
        BartNDArray array3 = array.concatenate(1, array2);
        for (int i = 0; i < array.shape(0); i++)
            for (int j = 0; j < array.shape(1); j++)
                for (int k = 0; k < array.shape(2); k++)
                    assertEquals(array.get(i, j, k), array3.get(i, j, k));
        for (int i = 0; i < array2.shape(0); i++)
            for (int j = array.shape(1); j < array.shape(1) + array2.shape(1); j++)
                for (int k = 0; k < array2.shape(2); k++)
                    assertEquals(new Complex(1, 0), array3.get(i, j, k));
    }

    @Test
    void testConcatenateMultiple() {
        BartNDArray array2 = array.copy().fill(1).slice(":", "1:3", ":");
        BartNDArray array3 = new BartComplexFloatNDArray(new int[]{3, 4, 4}).permuteDims(2, 1, 0);
        BartNDArray array4 = new BartComplexFloatNDArray(new int[]{12}).fill(new Complex(2, -2)).reshape(4, 1, 3);
        BartNDArray array5 = array.concatenate(1, array2, array3, array4);
        int start = 0;
        int end = array.shape(1);
        for (int i = 0; i < array.shape(0); i++)
            for (int j = start; j < end; j++)
                for (int k = 0; k < array.shape(2); k++)
                    assertEquals(array.get(i, j, k), array5.get(i, j, k));
        start = end;
        end += array2.shape(1);
        for (int i = 0; i < array2.shape(0); i++)
            for (int j = start; j < end; j++)
                for (int k = 0; k < array2.shape(2); k++)
                    assertEquals(new Complex(1, 0), array5.get(i, j, k));
        start = end;
        end += array3.shape(1);
        for (int i = 0; i < array2.shape(0); i++)
            for (int j = start; j < end; j++)
                for (int k = 0; k < array2.shape(2); k++)
                    assertEquals(new Complex(0, 0), array5.get(i, j, k));
        start = end;
        end += array4.shape(1);
        for (int i = 0; i < array2.shape(0); i++)
            for (int j = start; j < end; j++)
                for (int k = 0; k < array2.shape(2); k++)
                    assertEquals(new Complex(2, -2), array5.get(i, j, k));
    }

    @Test
    void testReal() {
        NDArray<Float> real = array.real();
        array.streamLinearIndices()
            .forEach(i -> assertEquals(array.get(i).getReal(), real.get(i).doubleValue()));
    }

    @Test
    void testImag() {
        NDArray<Float> imag = array.imaginary();
        array.streamLinearIndices()
            .forEach(i -> assertEquals(array.get(i).getImaginary(), imag.get(i).doubleValue()));
    }

    @Test
    void testAbs() {
        NDArray<Float> abs = array.abs();
        array.streamLinearIndices()
            .forEach(i -> assertTrue(array.get(i).abs() - abs.get(i) < 1e-5));
    }

    @Test
    void testAngle() {
        NDArray<Float> argument = array.argument();
        array.streamLinearIndices()
            .forEach(i -> assertTrue(array.get(i).getArgument() - argument.get(i) < 1e-5));
    }
}
