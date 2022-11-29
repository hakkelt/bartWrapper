package io.github.hakkelt.bartwrapper;

import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.function.BiFunction;
import java.util.function.Function;
import java.util.function.IntFunction;
import java.util.function.UnaryOperator;
import java.util.stream.Collector;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

import org.apache.commons.math3.complex.Complex;

import io.github.hakkelt.ndarrays.AbstractComplexNDArray;
import io.github.hakkelt.ndarrays.ComplexNDArray;
import io.github.hakkelt.ndarrays.NDArray;
import io.github.hakkelt.ndarrays.NDArrayUtils;
import io.github.hakkelt.ndarrays.basic.BasicFloatNDArray;
import io.github.hakkelt.ndarrays.internal.ComplexNDArrayCollector;

/**
 * Implementation for the NDArray of float (single-precision, 32 bit floating point) values
 * that is compatible with arrays expected by BART. It also contains some utility functions to
 * help dealing with the 16 dimensions used by BART.
 */
public class BartComplexFloatNDArray extends AbstractComplexNDArray<Float> implements BartNDArray {
    protected float[] data;
    protected BartDimsEnum[] bartDims = null;

    protected BartComplexFloatNDArray() {}

    /**
     * Simple constructor that defines only the shape of the NDArray and fills it with zeros.
     * 
     * @param dims dimensions / shape of the NDArray
     */
    public BartComplexFloatNDArray(int... dims) {
        baseConstuctor(dims);
        this.data = new float[length() * 2];
    }

    /**
     * Copy constructor.
     * 
     * @param array NDArray from which entries are copied from.
     */
    public BartComplexFloatNDArray(NDArray<?> array) {
        baseConstuctor(array.shape());
        this.data = new float[length() * 2];
        copyFrom(array);
        if (array instanceof BartNDArray && ((BartNDArray)array).areBartDimsSpecified())
            bartDims = ((BartNDArray)array).getBartDims();
    }

    /**
     * Copy constructor.
     * 
     * @param real NDArray from which real part of the new array is copied from.
     * @param imag NDArray from which imaginary part of the new array is copied from.
     */
    public BartComplexFloatNDArray(NDArray<? extends Number> real, NDArray<? extends Number> imag) {
        baseConstuctor(real.shape());
        this.data = new float[length() * 2];
        copyFrom(real, imag);
    }

    /**
     * Factory method that creates an NDArray from a list or 1D array of float values.
     * 
     * @param array a list or 1D array of float values from which a SimpleITKFloat32NDArray is created.
     * @return an NDArray created from a list or 1D array of float values
     */
    public static BartComplexFloatNDArray of(float... array) {
        return (BartComplexFloatNDArray)new BartComplexFloatNDArray(array.length).copyFrom(array);
    }
    
    /**
     * Factory method that creates an NDArray from a list or 1D array of double values.
     * 
     * @param array a list or 1D array of double values from which a SimpleITKFloat32NDArray is created.
     * @return an NDArray created from a list or 1D array of double values
     */
    public static BartComplexFloatNDArray of(double... array) {
        return (BartComplexFloatNDArray)new BartComplexFloatNDArray(array.length).copyFrom(array);
    }
    
    /**
     * Factory method that creates an NDArray from a list or 1D array of byte values.
     * 
     * @param array a list or 1D array of byte values from which a SimpleITKFloat32NDArray is created.
     * @return an NDArray created from a list or 1D array of byte values
     */
    public static BartComplexFloatNDArray of(byte... array) {
        return (BartComplexFloatNDArray)new BartComplexFloatNDArray(array.length).copyFrom(array);
    }
    
    /**
     * Factory method that creates an NDArray from a list or 1D array of short values.
     * 
     * @param array a list or 1D array of short values from which a SimpleITKFloat32NDArray is created.
     * @return an NDArray created from a list or 1D array of short values
     */
    public static BartComplexFloatNDArray of(short... array) {
        return (BartComplexFloatNDArray)new BartComplexFloatNDArray(array.length).copyFrom(array);
    }
    
    /**
     * Factory method that creates an NDArray from a list or 1D array of int values.
     * 
     * @param array a list or 1D array of int values from which a SimpleITKFloat32NDArray is created.
     * @return an NDArray created from a list or 1D array of int values
     */
    public static BartComplexFloatNDArray of(int... array) {
        return (BartComplexFloatNDArray)new BartComplexFloatNDArray(array.length).copyFrom(array);
    }
    
    /**
     * Factory method that creates an NDArray from a list or 1D array of long values.
     * 
     * @param array a list or 1D array of long values from which a SimpleITKFloat32NDArray is created.
     * @return an NDArray created from a list or 1D array of long values
     */
    public static BartComplexFloatNDArray of(long... array) {
        return (BartComplexFloatNDArray)new BartComplexFloatNDArray(array.length).copyFrom(array);
    }
    
    /**
     * Factory method that creates an NDArray from a multi-dimensional array of numeric values (including Complex type).
     * 
     * @param realOrComplex a multi-dimensional array of numeric values (including Complex type) 
     * from which a SimpleITKComplexFloat32NDArray is created.
     * @return an NDArray created from a multi-dimensional array of numeric values
     */
    public static BartComplexFloatNDArray of(Object[] realOrComplex) {
        return (BartComplexFloatNDArray)new BartComplexFloatNDArray(NDArrayUtils.computeDims(realOrComplex)).copyFrom(realOrComplex);
    }
    
    /**
     * Factory method that creates a BartNDArray from two 1D array of float values.
     * 
     * @param real a 1D array of float values from which the real part of the created SimpleITKComplexFloat32NDArray is read.
     * @param imag a 1D array of float values from which the imaginary part of the created SimpleITKComplexFloat32NDArray is read.
     * @return a BartNDArray created from the two 1D array of float values
     */
    public static BartComplexFloatNDArray of(float[] real, float[] imag) {
        return (BartComplexFloatNDArray)new BartComplexFloatNDArray(real.length).copyFrom(real, imag);
    }
    
    /**
     * Factory method that creates a BartNDArray from two 1D array of double values.
     * 
     * @param real a 1D array of double values from which the real part of the created SimpleITKComplexFloat32NDArray is read.
     * @param imag a 1D array of double values from which the imaginary part of the created SimpleITKComplexFloat32NDArray is read.
     * @return a BartNDArray created from the two 1D array of double values
     */
    public static BartComplexFloatNDArray of(double[] real, double[] imag) {
        return (BartComplexFloatNDArray)new BartComplexFloatNDArray(real.length).copyFrom(real, imag);
    }
    
    /**
     * Factory method that creates a BartNDArray from two 1D array of byte values.
     * 
     * @param real a 1D array of byte values from which the real part of the created SimpleITKComplexFloat32NDArray is read.
     * @param imag a 1D array of byte values from which the imaginary part of the created SimpleITKComplexFloat32NDArray is read.
     * @return a BartNDArray created from the two 1D array of byte values
     */
    public static BartComplexFloatNDArray of(byte[] real, byte[] imag) {
        return (BartComplexFloatNDArray)new BartComplexFloatNDArray(real.length).copyFrom(real, imag);
    }
    
    /**
     * Factory method that creates a BartNDArray from a list or 1D array of short values.
     * 
     * @param real a 1D array of short values from which the real part of the created SimpleITKComplexFloat32NDArray is read.
     * @param imag a 1D array of short values from which the imaginary part of the created SimpleITKComplexFloat32NDArray is read.
     * @return a BartNDArray created from the two 1D array of short values
     */
    public static BartComplexFloatNDArray of(short[] real, short[] imag) {
        return (BartComplexFloatNDArray)new BartComplexFloatNDArray(real.length).copyFrom(real, imag);
    }
    
    /**
     * Factory method that creates a BartNDArray from two 1D array of int values.
     * 
     * @param real a 1D array of int values from which the real part of the created SimpleITKComplexFloat32NDArray is read.
     * @param imag a 1D array of int values from which the imaginary part of the created SimpleITKComplexFloat32NDArray is read.
     * @return a BartNDArray created from the two 1D array of int values
     */
    public static BartComplexFloatNDArray of(int[] real, int[] imag) {
        return (BartComplexFloatNDArray)new BartComplexFloatNDArray(real.length).copyFrom(real, imag);
    }
    
    /**
     * Factory method that creates a BartNDArray from two 1D array of long values.
     * 
     * @param real a 1D array of long values from which the real part of the created SimpleITKComplexFloat32NDArray is read.
     * @param imag a 1D array of long values from which the imaginary part of the created SimpleITKComplexFloat32NDArray is read.
     * @return a BartNDArray created from the two 1D array of long values
     */
    public static BartComplexFloatNDArray of(long[] real, long[] imag) {
        return (BartComplexFloatNDArray)new BartComplexFloatNDArray(real.length).copyFrom(real, imag);
    }
    
    /**
     * Factory method that creates a BartNDArray from two multi-dimensional arrays of numeric values.
     * 
     * @param real a multi-dimensional array of numeric values from which the real part of the created SimpleITKComplexFloat32NDArray is read.
     * @param imag a multi-dimensional array of numeric values from which the imaginary part of the created SimpleITKComplexFloat32NDArray is read.
     * @return a BartNDArray created from the two multi-dimensional arrays of numeric values
     */
    public static BartComplexFloatNDArray of(Object[] real, Object[] imag) {
        return (BartComplexFloatNDArray)new BartComplexFloatNDArray(NDArrayUtils.computeDims(real)).copyFrom(real, imag);
    }

    @Override
    public BartComplexFloatNDArray copyFrom(NDArray<?> array) {
        if (array instanceof BartComplexFloatNDArray) {
            NDArrayUtils.checkShapeCompatibility(this, array.shape());
            data = ((BartComplexFloatNDArray) array).data.clone();
        } else
            super.copyFrom(array);
        return this;
    }

    @Override
    public BartNDArray similar() {
        return new BartComplexFloatNDArray(shape);
    }

    @Override
    public BartNDArray copy() {
        return new BartComplexFloatNDArray(this);
    }

    @Override
    public Float getRealUnchecked(int linearIndex) {
        return data[linearIndex * 2];
    }

    @Override
    public Float getRealUnchecked(int... indices) {
        return getRealUncheckedDefault(indices);
    }

    @Override
    public Float getImagUnchecked(int linearIndex) {
        return data[linearIndex * 2 + 1];
    }

    @Override
    public Float getImagUnchecked(int... indices) {
        return getImagUncheckedDefault(indices);
    }

    @Override
    protected Complex getUnchecked(int linearIndex) {
        return new Complex(getRealUnchecked(linearIndex), getImagUnchecked(linearIndex));
    }

    @Override
    protected Complex getUnchecked(int... indices) {
        return getUncheckedDefault(indices);
    }

    @Override
    protected void setUnchecked(Complex value, int linearIndex) {
        setRealUnchecked((float) value.getReal(), linearIndex);
        setImagUnchecked((float) value.getImaginary(), linearIndex);
    }

    @Override
    protected void setUnchecked(Complex value, int... indices) {
        setUncheckedDefault(value, indices);
    }

    @Override
    protected void setRealUnchecked(Float value, int linearIndex) {
        data[linearIndex * 2] = value;
    }

    @Override
    protected void setRealUnchecked(Float value, int... indices) {
        setRealUncheckedDefault(value, indices);
    }

    @Override
    protected void setImagUnchecked(Float value, int linearIndex) {
        data[linearIndex * 2 + 1] = value;
    }

    @Override
    protected void setImagUnchecked(Float value, int... indices) {
        setImagUncheckedDefault(value, indices);
    }

    public static Collector<Object, List<Object>, NDArray<Complex>> getCollector(int... dims) {
        return new ComplexNDArrayCollector<>(new BartComplexFloatNDArray(dims));
    }

    public boolean areBartDimsSpecified() {
        return bartDims != null;
    }
    
    public BartDimsEnum[] getBartDims() {
        if (bartDims == null)
            throw new UnsupportedOperationException(BartErrors.UNINITIALIZED_BART_DIMS);
        return bartDims;
    }
    
    public void setBartDims(BartDimsEnum... bartDims) {
        if (bartDims.length != ndim())
            throw new IllegalArgumentException(String.format(BartErrors.SET_BART_DIMS_SIZE_MISMATCH, ndim()));
        int[] intBartDims = IntStream.range(0, bartDims.length).map(i -> bartDims[i].ordinal()).toArray();
        if (hasDuplicates(intBartDims))
            throw new IllegalArgumentException(BartErrors.SET_BART_DIMS_DUPLICATES);
        this.bartDims = bartDims;
    }

    protected static boolean hasDuplicates(int[] array) {
        List<Integer> set = IntStream.of(array).boxed().collect(Collectors.toList());
        return IntStream.of(array).anyMatch(num -> Collections.frequency(set, num) > 1);
    }

    protected BartComplexFloatNDArray createNewNDArrayOfSameTypeAsMe(int... dims) {
        return new BartComplexFloatNDArray(dims);
    }

    protected NDArray<Float> createNewRealNDArrayOfSameTypeAsMe(int... dims) {
        return new BasicFloatNDArray(dims);
    }

    protected boolean areBartDimsEqual(BartNDArray array) {
        return Arrays.equals(bartDims, array.getBartDims());
    }

    @Override
    public boolean equals(Object obj) {
        if (obj instanceof BartNDArray && areBartDimsSpecified() != ((BartNDArray)obj).areBartDimsSpecified())
            return false;
        if (obj instanceof BartNDArray && areBartDimsSpecified() && !areBartDimsEqual((BartNDArray)obj))
            return false;
        if (!(obj instanceof BartNDArray) && areBartDimsSpecified())
            return false;
        if (obj instanceof BartComplexFloatNDArray)
            return Arrays.equals(data, ((BartComplexFloatNDArray) obj).data);
        return super.equals(obj);
    }

    @Override
    public int hashCode() {
        throw new UnsupportedOperationException();
    }

    @Override
    public String getNamePrefix() {
        return "bart";
    }
    
    @Override
    public BartNDArray apply(UnaryOperator<Complex> func) {
        super.apply(func);
        return this;
    }
    
    @Override
    public BartNDArray applyWithLinearIndices(BiFunction<Complex, Integer, Complex> func) {
        super.applyWithLinearIndices(func);
        return this;
    }
    
    @Override
    public BartNDArray applyWithCartesianIndices(BiFunction<Complex, int[], Complex> func) {
        super.applyWithCartesianIndices(func);
        return this;
    }
    
    @Override
    public BartNDArray fillUsingLinearIndices(IntFunction<Complex> func) {
        super.fillUsingLinearIndices(func);
        return this;
    }
    
    @Override
    public BartNDArray fillUsingCartesianIndices(Function<int[], Complex> func) {
        super.fillUsingCartesianIndices(func);
        return this;
    }

    @Override
    public BartNDArray applyOnComplexSlices(BiFunction<ComplexNDArray<Float>,int[],NDArray<?>> func, int... iterationDims) {
        super.applyOnComplexSlices(func, iterationDims);
        return this;
    }

    @Override
    public BartNDArray mapOnComplexSlices(BiFunction<ComplexNDArray<Float>,int[],NDArray<?>> func, int... iterationDims) {
        BartNDArray newInstance = copy();
        newInstance.applyOnComplexSlices(func, iterationDims);
        return newInstance;
    }
    
}

