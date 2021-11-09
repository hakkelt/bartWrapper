package io.github.hakkelt.bartwrapper;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;
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

import io.github.hakkelt.ndarrays.AbstractComplexNDArray;
import io.github.hakkelt.ndarrays.NDArray;
import io.github.hakkelt.ndarrays.NDArrayUtils;
import io.github.hakkelt.ndarrays.basic.BasicFloatNDArray;
import io.github.hakkelt.ndarrays.internal.ComplexNDArrayCollector;

import org.apache.commons.math3.complex.Complex;

/**
 * Reference implementation for the NDArray of float (single-precision, 32 bit floating point) values.
 */
public class BartFloatNDArray extends AbstractComplexNDArray<Float> implements BartNDArray {
    protected ByteBuffer byteBuffer;
    protected FloatBuffer floatBuffer;
    protected BartDimsEnum[] bartDims = null;


    protected BartFloatNDArray() {}

    /**
     * Simple constructor that defines only the shape of the NDArray and fills it with zeros.
     * 
     * @param dims dimensions / shape of the NDArray
     */
    public BartFloatNDArray(int... dims) {
        baseConstuctor(dims);
        byteBuffer = ByteBuffer.allocateDirect(dataLength * Float.BYTES * 2);
        byteBuffer.order(ByteOrder.LITTLE_ENDIAN);
        floatBuffer = byteBuffer.asFloatBuffer();
    }

    /**
     * Wrapper constructor.
     * 
     * Please note that changes to this wrapper will be reflected in the wrapped buffer and vice versa.
     * 
     * @param buffer ByteBuffer to be wrapped
     * @param dims shape of the BartNDArray to be created
     */
    public BartFloatNDArray(ByteBuffer buffer, int... dims) {
        baseConstuctor(dims);
        byteBuffer = buffer;
        if (byteBuffer.order() != ByteOrder.LITTLE_ENDIAN)
            throw new IllegalArgumentException(BartErrors.BYTE_ORDER_IS_NOT_LITTLE_ENDIAN);
        if (!byteBuffer.isDirect())
            throw new IllegalArgumentException(BartErrors.BYTE_BUFFER_IS_NOT_DIRECT);
        floatBuffer = byteBuffer.asFloatBuffer();
    }

    /**
     * Copy constructor.
     * 
     * @param array NDArray from which entries are copied from.
     */
    public BartFloatNDArray(NDArray<?> array) {
        baseConstuctor(array.shape());
        byteBuffer = ByteBuffer.allocateDirect(dataLength * Float.BYTES * 2);
        byteBuffer.order(ByteOrder.LITTLE_ENDIAN);
        floatBuffer = byteBuffer.asFloatBuffer();
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
    public BartFloatNDArray(NDArray<? extends Number> real, NDArray<? extends Number> imag) {
        baseConstuctor(real.shape());
        byteBuffer = ByteBuffer.allocateDirect(dataLength * Float.BYTES * 2);
        byteBuffer.order(ByteOrder.LITTLE_ENDIAN);
        floatBuffer = byteBuffer.asFloatBuffer();
        copyFrom(real, imag);
    }

    /**
     * Factory method that creates a new NDArray and copies all values from the supplied source.
     * 
     * @param source ByteBuffer from which values are read from
     * @param dims shape of the BartNDArray to be created
     * @return a BartNDArray of size dims filled with values from source ByteBuffer
     */
    public static BartFloatNDArray of(ByteBuffer source, int... dims) {
        return (BartFloatNDArray)new BartFloatNDArray(source, dims).copy();
    }

    /**
     * Factory method that creates an NDArray from a list or 1D array of float values.
     * 
     * @param array a list or 1D array of float values from which a SimpleITKFloat32NDArray is created.
     * @return an NDArray created from a list or 1D array of float values
     */
    public static BartFloatNDArray of(float... array) {
        return (BartFloatNDArray)new BartFloatNDArray(array.length).copyFrom(array);
    }
    
    /**
     * Factory method that creates an NDArray from a list or 1D array of double values.
     * 
     * @param array a list or 1D array of double values from which a SimpleITKFloat32NDArray is created.
     * @return an NDArray created from a list or 1D array of double values
     */
    public static BartFloatNDArray of(double... array) {
        return (BartFloatNDArray)new BartFloatNDArray(array.length).copyFrom(array);
    }
    
    /**
     * Factory method that creates an NDArray from a list or 1D array of byte values.
     * 
     * @param array a list or 1D array of byte values from which a SimpleITKFloat32NDArray is created.
     * @return an NDArray created from a list or 1D array of byte values
     */
    public static BartFloatNDArray of(byte... array) {
        return (BartFloatNDArray)new BartFloatNDArray(array.length).copyFrom(array);
    }
    
    /**
     * Factory method that creates an NDArray from a list or 1D array of short values.
     * 
     * @param array a list or 1D array of short values from which a SimpleITKFloat32NDArray is created.
     * @return an NDArray created from a list or 1D array of short values
     */
    public static BartFloatNDArray of(short... array) {
        return (BartFloatNDArray)new BartFloatNDArray(array.length).copyFrom(array);
    }
    
    /**
     * Factory method that creates an NDArray from a list or 1D array of int values.
     * 
     * @param array a list or 1D array of int values from which a SimpleITKFloat32NDArray is created.
     * @return an NDArray created from a list or 1D array of int values
     */
    public static BartFloatNDArray of(int... array) {
        return (BartFloatNDArray)new BartFloatNDArray(array.length).copyFrom(array);
    }
    
    /**
     * Factory method that creates an NDArray from a list or 1D array of long values.
     * 
     * @param array a list or 1D array of long values from which a SimpleITKFloat32NDArray is created.
     * @return an NDArray created from a list or 1D array of long values
     */
    public static BartFloatNDArray of(long... array) {
        return (BartFloatNDArray)new BartFloatNDArray(array.length).copyFrom(array);
    }
    
    /**
     * Factory method that creates an NDArray from a multi-dimensional array of numeric values (including Complex type).
     * 
     * @param realOrComplex a multi-dimensional array of numeric values (including Complex type) 
     * from which a SimpleITKComplexFloat32NDArray is created.
     * @return an NDArray created from a multi-dimensional array of numeric values
     */
    public static BartFloatNDArray of(Object[] realOrComplex) {
        return (BartFloatNDArray)new BartFloatNDArray(NDArrayUtils.computeDims(realOrComplex)).copyFrom(realOrComplex);
    }
    
    /**
     * Factory method that creates a BartNDArray from two 1D array of float values.
     * 
     * @param real a 1D array of float values from which the real part of the created SimpleITKComplexFloat32NDArray is read.
     * @param imag a 1D array of float values from which the imaginary part of the created SimpleITKComplexFloat32NDArray is read.
     * @return a BartNDArray created from the two 1D array of float values
     */
    public static BartFloatNDArray of(float[] real, float[] imag) {
        return (BartFloatNDArray)new BartFloatNDArray(real.length).copyFrom(real, imag);
    }
    
    /**
     * Factory method that creates a BartNDArray from two 1D array of double values.
     * 
     * @param real a 1D array of double values from which the real part of the created SimpleITKComplexFloat32NDArray is read.
     * @param imag a 1D array of double values from which the imaginary part of the created SimpleITKComplexFloat32NDArray is read.
     * @return a BartNDArray created from the two 1D array of double values
     */
    public static BartFloatNDArray of(double[] real, double[] imag) {
        return (BartFloatNDArray)new BartFloatNDArray(real.length).copyFrom(real, imag);
    }
    
    /**
     * Factory method that creates a BartNDArray from two 1D array of byte values.
     * 
     * @param real a 1D array of byte values from which the real part of the created SimpleITKComplexFloat32NDArray is read.
     * @param imag a 1D array of byte values from which the imaginary part of the created SimpleITKComplexFloat32NDArray is read.
     * @return a BartNDArray created from the two 1D array of byte values
     */
    public static BartFloatNDArray of(byte[] real, byte[] imag) {
        return (BartFloatNDArray)new BartFloatNDArray(real.length).copyFrom(real, imag);
    }
    
    /**
     * Factory method that creates a BartNDArray from a list or 1D array of short values.
     * 
     * @param real a 1D array of short values from which the real part of the created SimpleITKComplexFloat32NDArray is read.
     * @param imag a 1D array of short values from which the imaginary part of the created SimpleITKComplexFloat32NDArray is read.
     * @return a BartNDArray created from the two 1D array of short values
     */
    public static BartFloatNDArray of(short[] real, short[] imag) {
        return (BartFloatNDArray)new BartFloatNDArray(real.length).copyFrom(real, imag);
    }
    
    /**
     * Factory method that creates a BartNDArray from two 1D array of int values.
     * 
     * @param real a 1D array of int values from which the real part of the created SimpleITKComplexFloat32NDArray is read.
     * @param imag a 1D array of int values from which the imaginary part of the created SimpleITKComplexFloat32NDArray is read.
     * @return a BartNDArray created from the two 1D array of int values
     */
    public static BartFloatNDArray of(int[] real, int[] imag) {
        return (BartFloatNDArray)new BartFloatNDArray(real.length).copyFrom(real, imag);
    }
    
    /**
     * Factory method that creates a BartNDArray from two 1D array of long values.
     * 
     * @param real a 1D array of long values from which the real part of the created SimpleITKComplexFloat32NDArray is read.
     * @param imag a 1D array of long values from which the imaginary part of the created SimpleITKComplexFloat32NDArray is read.
     * @return a BartNDArray created from the two 1D array of long values
     */
    public static BartFloatNDArray of(long[] real, long[] imag) {
        return (BartFloatNDArray)new BartFloatNDArray(real.length).copyFrom(real, imag);
    }
    
    /**
     * Factory method that creates a BartNDArray from two multi-dimensional arrays of numeric values.
     * 
     * @param real a multi-dimensional array of numeric values from which the real part of the created SimpleITKComplexFloat32NDArray is read.
     * @param imag a multi-dimensional array of numeric values from which the imaginary part of the created SimpleITKComplexFloat32NDArray is read.
     * @return a BartNDArray created from the two multi-dimensional arrays of numeric values
     */
    public static BartFloatNDArray of(Object[] real, Object[] imag) {
        return (BartFloatNDArray)new BartFloatNDArray(NDArrayUtils.computeDims(real)).copyFrom(real, imag);
    }

    @Override
    public BartFloatNDArray copyFrom(NDArray<?> array) {
        if (array instanceof BartFloatNDArray) {
            byteBuffer.rewind().put(((BartFloatNDArray)array).byteBuffer.rewind());
            if (((BartFloatNDArray)array).bartDims != null)
                bartDims = ((BartFloatNDArray)array).bartDims.clone();
        } else if (array instanceof BartNDArrayReshapeView &&
                ((BartNDArrayReshapeView)array).getParent() instanceof BartFloatNDArray) {
            byteBuffer.rewind().put(((BartFloatNDArray)((BartNDArrayReshapeView)array).getParent()).byteBuffer.rewind());
        } else {
            super.copyFrom(array);
        }
        return this;
    }

    @Override
    public BartNDArray similar() {
        return new BartFloatNDArray(shape);
    }

    @Override
    public BartNDArray copy() {
        return new BartFloatNDArray(this);
    }

    @Override
    public Float getRealUnchecked(int linearIndex) {
        return floatBuffer.get(linearIndex * 2);
    }

    @Override
    public Float getRealUnchecked(int... indices) {
        return getRealUncheckedDefault(indices);
    }

    @Override
    public Float getImagUnchecked(int linearIndex) {
        return floatBuffer.get(linearIndex * 2 + 1);
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
        floatBuffer.put(linearIndex * 2, value);
    }

    @Override
    protected void setRealUnchecked(Float value, int... indices) {
        setRealUncheckedDefault(value, indices);
    }

    @Override
    protected void setImagUnchecked(Float value, int linearIndex) {
        floatBuffer.put(linearIndex * 2 + 1, value);
    }

    @Override
    protected void setImagUnchecked(Float value, int... indices) {
        setImagUncheckedDefault(value, indices);
    }

    public static Collector<Object, List<Object>, NDArray<Complex>> getCollector(int... dims) {
        return new ComplexNDArrayCollector<>(new BartFloatNDArray(dims));
    }

    public ByteBuffer getByteBuffer() {
        return byteBuffer;
    }

    public FloatBuffer getFloatBuffer() {
        return floatBuffer;
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

    protected BartFloatNDArray createNewNDArrayOfSameTypeAsMe(int... dims) {
        return new BartFloatNDArray(dims);
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
        if (obj instanceof BartFloatNDArray)
            return ((BartFloatNDArray)obj).byteBuffer.rewind().equals(byteBuffer.rewind());
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
    
}

