package io.github.hakkelt.bartwrapper;

import java.util.function.BiConsumer;
import java.util.function.BiFunction;
import java.util.function.BiPredicate;
import java.util.function.Function;
import java.util.function.IntFunction;
import java.util.function.Predicate;
import java.util.function.UnaryOperator;

import org.apache.commons.math3.complex.Complex;

import io.github.hakkelt.ndarrays.ComplexNDArray;
import io.github.hakkelt.ndarrays.NDArray;
import io.github.hakkelt.ndarrays.internal.ComplexNDArrayMaskView;

/**
 * A view for a ComplexNDArray that selects values based on a specified mask.
 * When mask(...) is called for a ComplexNDArray, an instance of this class is returned.
 */
public class BartNDArrayMaskView extends ComplexNDArrayMaskView<Float> implements BartNDArray, BartNDArrayView {
    
    public BartNDArrayMaskView(BartNDArray parent, NDArray<?> mask, boolean isInverse) {
        super(parent instanceof BartNDArrayMaskView ? (BartNDArrayMaskView)parent : parent, mask, isInverse);
    } 
    
    public BartNDArrayMaskView(BartNDArray parent, Predicate<Complex> func) {
        super(parent instanceof BartNDArrayMaskView ? (BartNDArrayMaskView)parent : parent, func);
    }
    
    public BartNDArrayMaskView(BartNDArray parent, BiPredicate<Complex,?> func, boolean withLinearIndices) {
        super(parent instanceof BartNDArrayMaskView ? (BartNDArrayMaskView)parent : parent, func, withLinearIndices);
    }

    @Override
    public BartNDArray similar() {
        return createNewNDArrayOfSameTypeAsMe(shape);
    }

    @Override
    public BartNDArray copy() {
        return similar().copyFrom(this);
    }

    @Override
    public boolean areBartDimsSpecified() {
        return false;
    }
    
    public BartDimsEnum[] getBartDims() {
        throw new UnsupportedOperationException(BartErrors.UNINITIALIZED_BART_DIMS);
    }
    
    public void setBartDims(BartDimsEnum... bartDims) {
        throw new UnsupportedOperationException(BartErrors.CANNOT_SET_BART_DIMS_ON_VIEW);
    }

    @Override
    public BartNDArray createNewNDArrayOfSameTypeAsMe(int... shape) {
        if (parent instanceof BartComplexFloatNDArray)
            return ((BartComplexFloatNDArray)parent).createNewNDArrayOfSameTypeAsMe(shape);
        else
            return ((BartNDArrayView)parent).createNewNDArrayOfSameTypeAsMe(shape);
    }

    @Override
    public boolean equals(Object obj) {
        if (obj instanceof BartNDArray && ((BartNDArray)obj).areBartDimsSpecified())
            return false;
        return super.equals(obj);
    }

    @Override
    public int hashCode() {
        throw new UnsupportedOperationException();
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
    public BartNDArray fillUsingLinearIndices(IntFunction< Complex> func) {
        super.fillUsingLinearIndices(func);
        return this;
    }
    
    @Override
    public BartNDArray fillUsingCartesianIndices(Function<int[], Complex> func) {
        super.fillUsingCartesianIndices(func);
        return this;
    }

    @Override
    public BartNDArray applyOnComplexSlices(BiConsumer<ComplexNDArray<Float>,int[]> func, int... iterationDims) {
        super.applyOnComplexSlices(func, iterationDims);
        return this;
    }

    @Override
    public BartNDArray applyOnComplexSlices(BiFunction<ComplexNDArray<Float>,int[],NDArray<?>> func, int... iterationDims) {
        super.applyOnComplexSlices(func, iterationDims);
        return this;
    }
    
    @Override
    public BartNDArray mapOnComplexSlices(BiConsumer<ComplexNDArray<Float>,int[]> func, int... iterationDims) {
        BartNDArray newInstance = copy();
        newInstance.applyOnComplexSlices(func, iterationDims);
        return newInstance;
    }

    @Override
    public BartNDArray mapOnComplexSlices(BiFunction<ComplexNDArray<Float>,int[],NDArray<?>> func, int... iterationDims) {
        BartNDArray newInstance = copy();
        newInstance.applyOnComplexSlices(func, iterationDims);
        return newInstance;
    }
    
}
