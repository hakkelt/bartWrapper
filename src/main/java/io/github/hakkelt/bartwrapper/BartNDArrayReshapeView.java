package io.github.hakkelt.bartwrapper;

import java.util.function.BiFunction;
import java.util.function.Function;
import java.util.function.IntFunction;
import java.util.function.UnaryOperator;

import org.apache.commons.math3.complex.Complex;

import io.github.hakkelt.ndarrays.internal.ComplexNDArrayReshapeView;

/**
 * A view for a ComplexNDArray that changes the shape of the parent NDArray.
 * When reshape(...) is called for a ComplexNDArray, an instance of this class is returned.
 */
public class BartNDArrayReshapeView extends ComplexNDArrayReshapeView<Float> implements BartNDArray, BartNDArrayView {
    
    public BartNDArrayReshapeView(BartNDArray parent, int ...newShape) {
        super(parent, newShape);
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
        if (parent instanceof BartFloatNDArray)
            return ((BartFloatNDArray)parent).createNewNDArrayOfSameTypeAsMe(shape);
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
    
}
