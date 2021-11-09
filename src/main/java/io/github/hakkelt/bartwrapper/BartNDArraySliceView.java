package io.github.hakkelt.bartwrapper;

import java.util.Arrays;
import java.util.function.BiFunction;
import java.util.function.Function;
import java.util.function.IntFunction;
import java.util.function.UnaryOperator;
import java.util.stream.IntStream;

import org.apache.commons.math3.complex.Complex;

import io.github.hakkelt.ndarrays.Range;
import io.github.hakkelt.ndarrays.internal.ComplexNDArraySliceView;

/**
 * A view for a ComplexNDArray that slices the parent ComplexNDArray.
 * When slice(...) is called for a ComplexNDArray, an instance of this class is returned.
 */
public class BartNDArraySliceView extends ComplexNDArraySliceView<Float> implements BartNDArray, BartNDArrayView {
    
    public BartNDArraySliceView(BartNDArray parent, Range[] slicingExpressions) {
        super(parent, slicingExpressions);
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
        Range[] expr = slicingExpression.getExpressions();
        return IntStream.range(0, expr.length)
            .filter(i -> !expr[i].isScalar())
            .mapToObj(d -> bartDims[d])
            .toArray(BartDimsEnum[]::new);
    }
    
    public void setBartDims(BartDimsEnum... bartDims) {
        throw new UnsupportedOperationException(BartErrors.CANNOT_SET_BART_DIMS_ON_VIEW);
    }

    @Override
    public BartNDArray createNewNDArrayOfSameTypeAsMe(int... dims) {
        if (parent instanceof BartFloatNDArray)
            return ((BartFloatNDArray)parent).createNewNDArrayOfSameTypeAsMe(dims);
        else
            return ((BartNDArrayView)parent).createNewNDArrayOfSameTypeAsMe(dims);
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
    
}
