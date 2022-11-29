package io.github.hakkelt.bartwrapper;

import java.util.Arrays;
import java.util.function.BiFunction;
import java.util.function.Function;
import java.util.function.IntFunction;
import java.util.function.UnaryOperator;
import java.util.stream.IntStream;

import org.apache.commons.math3.complex.Complex;

import io.github.hakkelt.ndarrays.ComplexNDArray;
import io.github.hakkelt.ndarrays.NDArray;
import io.github.hakkelt.ndarrays.internal.ComplexNDArrayPermuteDimsView;

/**
 * A view for a ComplexNDArray that permutes the order of dimensions.
 * When permuteDims(...) is called for a ComplexNDArray, an instance of this class is returned.
 */
public class BartNDArrayPermuteDimsView extends ComplexNDArrayPermuteDimsView<Float> implements BartNDArray, BartNDArrayView {

    public BartNDArrayPermuteDimsView(BartNDArray parent, int ...dimsOrder) {
        super(parent, dimsOrder);
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
        return ((BartNDArray)parent).areBartDimsSpecified();
    }
    
    public BartDimsEnum[] getBartDims() {
        BartDimsEnum[] bartDims = ((BartNDArray)parent).getBartDims();
        return IntStream.of(dimsOrder).mapToObj(d -> bartDims[d]).toArray(BartDimsEnum[]::new);
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

    protected boolean areBartDimsEqual(BartNDArray array) {
        return Arrays.equals(getBartDims(), array.getBartDims());
    }

    @Override
    public boolean equals(Object obj) {
        if (obj instanceof BartNDArray && areBartDimsSpecified() != ((BartNDArray)obj).areBartDimsSpecified())
            return false;
        if (obj instanceof BartNDArray && areBartDimsSpecified() && !areBartDimsEqual((BartNDArray)obj))
            return false;
        if (!(obj instanceof BartNDArray) && areBartDimsSpecified())
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
