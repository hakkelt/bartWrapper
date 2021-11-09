package io.github.hakkelt.bartwrapper;

import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Set;
import java.util.function.BiFunction;
import java.util.function.BiPredicate;
import java.util.function.Function;
import java.util.function.IntFunction;
import java.util.function.Predicate;
import java.util.function.UnaryOperator;
import java.util.stream.Collectors;
import java.util.stream.IntStream;
import java.util.stream.Stream;

import org.apache.commons.math3.complex.Complex;

import io.github.hakkelt.ndarrays.internal.ArrayOperations;
import io.github.hakkelt.ndarrays.ComplexNDArray;
import io.github.hakkelt.ndarrays.internal.CopyFromOperations;
import io.github.hakkelt.ndarrays.internal.Errors;
import io.github.hakkelt.ndarrays.NDArray;
import io.github.hakkelt.ndarrays.Range;
import io.github.hakkelt.ndarrays.internal.ViewOperations;

public interface BartNDArray extends ComplexNDArray<Float> {

    @Override
    public default BartNDArray copyFrom(float[] array) {
        new CopyFromOperations<Complex,Float>().copyFrom(this, array);
        return this; 
    }

    @Override
    public default BartNDArray copyFrom(double[] array) {
        new CopyFromOperations<Complex,Float>().copyFrom(this, array);
        return this;
    }

    @Override
    public default BartNDArray copyFrom(byte[] array) {
        new CopyFromOperations<Complex,Float>().copyFrom(this, array);
        return this;
    }

    @Override
    public default BartNDArray copyFrom(short[] array) {
        new CopyFromOperations<Complex,Float>().copyFrom(this, array);
        return this;
    }

    @Override
    public default BartNDArray copyFrom(int[] array) {
        new CopyFromOperations<Complex,Float>().copyFrom(this, array);
        return this;
    }

    @Override
    public default BartNDArray copyFrom(long[] array) {
        new CopyFromOperations<Complex,Float>().copyFrom(this, array);
        return this;
    }

    @Override
    public default BartNDArray copyFrom(Object[] array) {
        new CopyFromOperations<Complex,Float>().copyFrom(this, array);
        return this;
    }

    @Override
    public default BartNDArray copyFrom(NDArray<?> array) {
        new CopyFromOperations<Complex,Float>().copyFrom(this, array);
        return this;
    }
    
    @Override
    public default BartNDArray copyFrom(float[] real, float[] imag) {
        new CopyFromOperations<Complex,Float>().copyFrom(this, real, imag);
        return this;
    }
    
    @Override
    public default BartNDArray copyFrom(double[] real, double[] imag) {
        new CopyFromOperations<Complex,Float>().copyFrom(this, real, imag);
        return this;
    }
    
    @Override
    public default BartNDArray copyFrom(byte[] real, byte[] imag) {
        new CopyFromOperations<Complex,Float>().copyFrom(this, real, imag);
        return this;
    }
    
    @Override
    public default BartNDArray copyFrom(short[] real, short[] imag) {
        new CopyFromOperations<Complex,Float>().copyFrom(this, real, imag);
        return this;
    }
    
    @Override
    public default BartNDArray copyFrom(int[] real, int[] imag) {
        new CopyFromOperations<Complex,Float>().copyFrom(this, real, imag);
        return this;
    }
    
    @Override
    public default BartNDArray copyFrom(long[] real, long[] imag) {
        new CopyFromOperations<Complex,Float>().copyFrom(this, real, imag);
        return this;
    }

    @Override
    public default BartNDArray copyFrom(Object[] real, Object[] imag) {
        new CopyFromOperations<Complex,Float>().copyFrom(this, real, imag);
        return this;
    }
    
    @Override
    public default BartNDArray copyFrom(NDArray<? extends Number> real, NDArray<? extends Number> imag) {
        new CopyFromOperations<Complex,Float>().copyFrom(this, real, imag);
        return this;
    }
    
    @Override
    public BartNDArray apply(UnaryOperator<Complex> func);
    
    @Override
    public BartNDArray applyWithLinearIndices(BiFunction<Complex, Integer, Complex> func);
    
    @Override
    public BartNDArray applyWithCartesianIndices(BiFunction<Complex, int[], Complex> func);
    
    @Override
    public BartNDArray fillUsingLinearIndices(IntFunction<Complex> func);
    
    @Override
    public BartNDArray fillUsingCartesianIndices(Function<int[], Complex> func);
    
    @Override
    public default BartNDArray map(UnaryOperator<Complex> func) {
        BartNDArray newInstance = copy();
        return newInstance.apply(func);
    }
    
    @Override
    public default BartNDArray mapWithLinearIndices(BiFunction<Complex, Integer, Complex> func) {
        BartNDArray newInstance = copy();
        newInstance.applyWithLinearIndices(func);
        return newInstance;
    }
    
    @Override
    public default BartNDArray mapWithCartesianIndices(BiFunction<Complex, int[], Complex> func) {
        BartNDArray newInstance = copy();
        newInstance.applyWithCartesianIndices(func);
        return newInstance;
    }

    @Override
    public default BartNDArray add(byte addend) {
        BartNDArray newInstance = copy();
        new ArrayOperations<Complex,Float>().addInplace(newInstance, addend);
        return newInstance;
    }

    @Override
    public default BartNDArray add(short addend) {
        BartNDArray newInstance = copy();
        new ArrayOperations<Complex,Float>().addInplace(newInstance, addend);
        return newInstance;
    }

    @Override
    public default BartNDArray add(int addend) {
        BartNDArray newInstance = copy();
        new ArrayOperations<Complex,Float>().addInplace(newInstance, addend);
        return newInstance;
    }

    @Override
    public default BartNDArray add(long addend) {
        BartNDArray newInstance = copy();
        new ArrayOperations<Complex,Float>().addInplace(newInstance, addend);
        return newInstance;
    }

    @Override
    public default BartNDArray add(float addend) {
        BartNDArray newInstance = copy();
        new ArrayOperations<Complex,Float>().addInplace(newInstance, addend);
        return newInstance;
    }

    @Override
    public default BartNDArray add(double addend) {
        BartNDArray newInstance = copy();
        new ArrayOperations<Complex,Float>().addInplace(newInstance, addend);
        return newInstance;
    }

    @Override
    public default BartNDArray add(NDArray<?> addend) {
        BartNDArray newInstance = copy();
        new ArrayOperations<Complex,Float>().addInplace(newInstance, addend);
        return newInstance;
    }

    @Override
    public default BartNDArray add(Object... addends) {
        BartNDArray newInstance = copy();
        new ArrayOperations<Complex,Float>().addInplace(newInstance, addends);
        return newInstance;
    }

    @Override
    public default BartNDArray addInplace(byte addend) {
        new ArrayOperations<Complex,Float>().addInplace(this, addend);
        return this;
    }

    @Override
    public default BartNDArray addInplace(short addend) {
        new ArrayOperations<Complex,Float>().addInplace(this, addend);
        return this;
    }

    @Override
    public default BartNDArray addInplace(int addend) {
        new ArrayOperations<Complex,Float>().addInplace(this, addend);
        return this;
    }

    @Override
    public default BartNDArray addInplace(long addend) {
        new ArrayOperations<Complex,Float>().addInplace(this, addend);
        return this;
    }

    @Override
    public default BartNDArray addInplace(float addend) {
        new ArrayOperations<Complex,Float>().addInplace(this, addend);
        return this;
    }

    @Override
    public default BartNDArray addInplace(double addend) {
        new ArrayOperations<Complex,Float>().addInplace(this, addend);
        return this;
    }

    @Override
    public default BartNDArray addInplace(NDArray<?> addend) {
        new ArrayOperations<Complex,Float>().addInplace(this, addend);
        return this;
    }

    @Override
    public default BartNDArray addInplace(Object... addends) {
        new ArrayOperations<Complex,Float>().addInplace(this, addends);
        return this;
    }

    @Override
    public default BartNDArray subtract(byte substrahend) {
        BartNDArray newInstance = copy();
        new ArrayOperations<Complex,Float>().subtractInplace(newInstance, substrahend);
        return newInstance;
    }

    @Override
    public default BartNDArray subtract(short substrahend) {
        BartNDArray newInstance = copy();
        new ArrayOperations<Complex,Float>().subtractInplace(newInstance, substrahend);
        return newInstance;
    }

    @Override
    public default BartNDArray subtract(int substrahend) {
        BartNDArray newInstance = copy();
        new ArrayOperations<Complex,Float>().subtractInplace(newInstance, substrahend);
        return newInstance;
    }

    @Override
    public default BartNDArray subtract(long substrahend) {
        BartNDArray newInstance = copy();
        new ArrayOperations<Complex,Float>().subtractInplace(newInstance, substrahend);
        return newInstance;
    }

    @Override
    public default BartNDArray subtract(float substrahend) {
        BartNDArray newInstance = copy();
        new ArrayOperations<Complex,Float>().subtractInplace(newInstance, substrahend);
        return newInstance;
    }

    @Override
    public default BartNDArray subtract(double substrahend) {
        BartNDArray newInstance = copy();
        new ArrayOperations<Complex,Float>().subtractInplace(newInstance, substrahend);
        return newInstance;
    }

    @Override
    public default BartNDArray subtract(NDArray<?> substrahend) {
        BartNDArray newInstance = copy();
        new ArrayOperations<Complex,Float>().subtractInplace(newInstance, substrahend);
        return newInstance;
    }

    @Override
    public default BartNDArray subtract(Object... substrahends) {
        BartNDArray newInstance = copy();
        new ArrayOperations<Complex,Float>().subtractInplace(newInstance, substrahends);
        return newInstance;
    }
    
    @Override
    public default BartNDArray subtractInplace(byte substrahend) {
        new ArrayOperations<Complex,Float>().subtractInplace(this, substrahend);
        return this;
    }
    
    @Override
    public default BartNDArray subtractInplace(short substrahend) {
        new ArrayOperations<Complex,Float>().subtractInplace(this, substrahend);
        return this;
    }
    
    @Override
    public default BartNDArray subtractInplace(int substrahend) {
        new ArrayOperations<Complex,Float>().subtractInplace(this, substrahend);
        return this;
    }
    
    @Override
    public default BartNDArray subtractInplace(long substrahend) {
        new ArrayOperations<Complex,Float>().subtractInplace(this, substrahend);
        return this;
    }
    
    @Override
    public default BartNDArray subtractInplace(float substrahend) {
        new ArrayOperations<Complex,Float>().subtractInplace(this, substrahend);
        return this;
    }
    
    @Override
    public default BartNDArray subtractInplace(double substrahend) {
        new ArrayOperations<Complex,Float>().subtractInplace(this, substrahend);
        return this;
    }
    
    @Override
    public default BartNDArray subtractInplace(NDArray<?> substrahend) {
        new ArrayOperations<Complex,Float>().subtractInplace(this, substrahend);
        return this;
    }
    
    @Override
    public default BartNDArray subtractInplace(Object... substrahends) {
        new ArrayOperations<Complex,Float>().subtractInplace(this, substrahends);
        return this;
    }

    @Override
    public default BartNDArray multiply(byte multiplicand) {
        BartNDArray newInstance = copy();
        new ArrayOperations<Complex,Float>().multiplyInplace(newInstance, multiplicand);
        return newInstance;
    }

    @Override
    public default BartNDArray multiply(short multiplicand) {
        BartNDArray newInstance = copy();
        new ArrayOperations<Complex,Float>().multiplyInplace(newInstance, multiplicand);
        return newInstance;
    }

    @Override
    public default BartNDArray multiply(int multiplicand) {
        BartNDArray newInstance = copy();
        new ArrayOperations<Complex,Float>().multiplyInplace(newInstance, multiplicand);
        return newInstance;
    }

    @Override
    public default BartNDArray multiply(long multiplicand) {
        BartNDArray newInstance = copy();
        new ArrayOperations<Complex,Float>().multiplyInplace(newInstance, multiplicand);
        return newInstance;
    }

    @Override
    public default BartNDArray multiply(float multiplicand) {
        BartNDArray newInstance = copy();
        new ArrayOperations<Complex,Float>().multiplyInplace(newInstance, multiplicand);
        return newInstance;
    }

    @Override
    public default BartNDArray multiply(double multiplicand) {
        BartNDArray newInstance = copy();
        new ArrayOperations<Complex,Float>().multiplyInplace(newInstance, multiplicand);
        return newInstance;
    }

    @Override
    public default BartNDArray multiply(NDArray<?> multiplicand) {
        BartNDArray newInstance = copy();
        new ArrayOperations<Complex,Float>().multiplyInplace(newInstance, multiplicand);
        return newInstance;
    }

    @Override
    public default BartNDArray multiply(Object... multiplicands) {
        BartNDArray newInstance = copy();
        new ArrayOperations<Complex,Float>().multiplyInplace(newInstance, multiplicands);
        return newInstance;
    }

    @Override
    public default BartNDArray multiplyInplace(byte multiplicand) {
        new ArrayOperations<Complex,Float>().multiplyInplace(this, multiplicand);
        return this;
    }

    @Override
    public default BartNDArray multiplyInplace(short multiplicand) {
        new ArrayOperations<Complex,Float>().multiplyInplace(this, multiplicand);
        return this;
    }

    @Override
    public default BartNDArray multiplyInplace(int multiplicand) {
        new ArrayOperations<Complex,Float>().multiplyInplace(this, multiplicand);
        return this;
    }

    @Override
    public default BartNDArray multiplyInplace(long multiplicand) {
        new ArrayOperations<Complex,Float>().multiplyInplace(this, multiplicand);
        return this;
    }

    @Override
    public default BartNDArray multiplyInplace(float multiplicand) {
        new ArrayOperations<Complex,Float>().multiplyInplace(this, multiplicand);
        return this;
    }

    @Override
    public default BartNDArray multiplyInplace(double multiplicand) {
        new ArrayOperations<Complex,Float>().multiplyInplace(this, multiplicand);
        return this;
    }

    @Override
    public default BartNDArray multiplyInplace(NDArray<?> multiplicand) {
        new ArrayOperations<Complex,Float>().multiplyInplace(this, multiplicand);
        return this;
    }

    @Override
    public default BartNDArray multiplyInplace(Object... multiplicands) {
        new ArrayOperations<Complex,Float>().multiplyInplace(this, multiplicands);
        return this;
    }

    @Override
    public default BartNDArray divide(byte divisor) {
        BartNDArray newInstance = copy();
        new ArrayOperations<Complex,Float>().divideInplace(newInstance, divisor);
        return newInstance;
    }

    @Override
    public default BartNDArray divide(short divisor) {
        BartNDArray newInstance = copy();
        new ArrayOperations<Complex,Float>().divideInplace(newInstance, divisor);
        return newInstance;
    }

    @Override
    public default BartNDArray divide(int divisor) {
        BartNDArray newInstance = copy();
        new ArrayOperations<Complex,Float>().divideInplace(newInstance, divisor);
        return newInstance;
    }

    @Override
    public default BartNDArray divide(long divisor) {
        BartNDArray newInstance = copy();
        new ArrayOperations<Complex,Float>().divideInplace(newInstance, divisor);
        return newInstance;
    }

    @Override
    public default BartNDArray divide(float divisor) {
        BartNDArray newInstance = copy();
        new ArrayOperations<Complex,Float>().divideInplace(newInstance, divisor);
        return newInstance;
    }

    @Override
    public default BartNDArray divide(double divisor) {
        BartNDArray newInstance = copy();
        new ArrayOperations<Complex,Float>().divideInplace(newInstance, divisor);
        return newInstance;
    }

    @Override
    public default BartNDArray divide(NDArray<?> divisor) {
        BartNDArray newInstance = copy();
        new ArrayOperations<Complex,Float>().divideInplace(newInstance, divisor);
        return newInstance;
    }

    @Override
    public default BartNDArray divide(Object... divisors) {
        BartNDArray newInstance = copy();
        new ArrayOperations<Complex,Float>().divideInplace(newInstance, divisors);
        return newInstance;
    }

    @Override
    public default BartNDArray divideInplace(byte divisor) {
        new ArrayOperations<Complex,Float>().divideInplace(this, divisor);
        return this;
    }

    @Override
    public default BartNDArray divideInplace(short divisor) {
        new ArrayOperations<Complex,Float>().divideInplace(this, divisor);
        return this;
    }

    @Override
    public default BartNDArray divideInplace(int divisor) {
        new ArrayOperations<Complex,Float>().divideInplace(this, divisor);
        return this;
    }

    @Override
    public default BartNDArray divideInplace(long divisor) {
        new ArrayOperations<Complex,Float>().divideInplace(this, divisor);
        return this;
    }

    @Override
    public default BartNDArray divideInplace(float divisor) {
        new ArrayOperations<Complex,Float>().divideInplace(this, divisor);
        return this;
    }

    @Override
    public default BartNDArray divideInplace(double divisor) {
        new ArrayOperations<Complex,Float>().divideInplace(this, divisor);
        return this;
    }

    @Override
    public default BartNDArray divideInplace(NDArray<?> divisor) {
        new ArrayOperations<Complex,Float>().divideInplace(this, divisor);
        return this;
    }

    @Override
    public default BartNDArray divideInplace(Object... divisors) {
        new ArrayOperations<Complex,Float>().divideInplace(this, divisors);
        return this;
    }

    @Override
    public default BartNDArray sum(int... selectedDims) {
        return (BartNDArray)new ArrayOperations<Complex,Float>().sum(this, selectedDims);
    }
    
    @Override
    public default BartNDArray fill(Complex value) {
        new ArrayOperations<Complex,Float>().fill(this, value);
        return this;
    }
    
    @Override
    public default BartNDArray fill(Float value) {
        new ArrayOperations<Complex,Float>().fill(this, value);
        return this;
    }

    @Override
    public default BartNDArray fill(double value) {
        new ArrayOperations<Complex,Float>().fill(this, value);
        return this;
    }

    @Override
    public default BartNDArray concatenate(int axis, NDArray<?> ...arrays) {
        return (BartNDArray)new ArrayOperations<Complex,Float>().concatenate(this, axis, arrays);
    }

    public BartNDArray similar();
    public BartNDArray copy();
    
    @Override
    public default BartNDArray slice(Object... slicingExpressions) {
        Range[] expressions = Range.parseExpressions(shape(), slicingExpressions);
        if (ViewOperations.isThisSlicingAnIdentityOperation(this, expressions)) return this;
        return new BartNDArraySliceView(this, expressions);
    }
    
    @Override
    public default BartNDArray mask(NDArray<?> mask) {
        BartNDArray view = new BartNDArrayMaskView(this, mask, false);
        return view.length() == length() ? reshape(length()) : view;
    }
    
    @Override
    public default BartNDArray mask(Predicate<Complex> func) {
        BartNDArray view = new BartNDArrayMaskView(this, func);
        return view.length() == length() ? reshape(length()) : view;
    }
    
    @Override
    public default BartNDArray maskWithLinearIndices(BiPredicate<Complex,Integer> func) {
        BartNDArray view = new BartNDArrayMaskView(this, func, true);
        return view.length() == length() ? reshape(length()) : view;
    }
    
    @Override
    public default BartNDArray maskWithCartesianIndices(BiPredicate<Complex,int[]> func) {
        BartNDArray view = new BartNDArrayMaskView(this, func, false);
        return view.length() == length() ? reshape(length()) : view;
    }
    
    @Override
    public default BartNDArray inverseMask(NDArray<?> mask) {
        BartNDArray view = new BartNDArrayMaskView(this, mask, true);
        return view.length() == length() ? reshape(length()) : view;
    }

    @Override
    public default BartNDArray permuteDims(int... permutation) {
        if (ViewOperations.isThisPermutationAnIdentityOperation(permutation)) return this;
        BartNDArrayPermuteDimsView view = new BartNDArrayPermuteDimsView(this, permutation);
        return IntStream.range(0, ndim()).allMatch(i -> i == view.getDimsOrder()[i]) ?
            (BartNDArray)view.getParent() :
            view;
    }

    @Override
    public default BartNDArray reshape(int... newShape) {
        if (Arrays.equals(shape(), newShape)) return this;
        if (this instanceof BartNDArrayReshapeView &&
                Arrays.equals(newShape, ((BartNDArrayReshapeView)this).getParent().shape()))
            return (BartNDArray)((BartNDArrayReshapeView)this).getParent();
        return new BartNDArrayReshapeView(this, newShape);
    }

    @Override
    public default BartNDArray selectDims(int... selectedDims) {
        ViewOperations.checkSelectedDims(ndim(), selectedDims);
        Set<Integer> set = ViewOperations.intArrayToSet(selectedDims);
        if (set.size() == ndim()) return this;
        return slice(ViewOperations.selectedDimsToSlicingExpression(shape(), set));
    }
    
    /** 
     * Returns a view that references this BartNDArray as parent,
     * skips all singleton dimensions not included in the parameter list,
     * and then changes the order of dimensions to match the order of dimensions in the parameter list.
     * 
     * <p>View: An NDArray that references the specified region its parent array.
     * All modifications in the parent array are reflected in the view, and vice versa.
     * 
     * <p>Singleton dimension: dimension of size 1.
     * 
     * <ul><li><b>Example:</b></li></ul>
     * 
     * <blockquote><pre>{@code 
BartNDArray array = new BartFloatNDArray(128, 1, 64).fill(new Complex(1,-1));
array.setBartDims(BartDimsEnum._01_PHS1, BartDimsEnum._00_READ, BartDimsEnum._10_TIME);
BartNDArray array2 = array.selectAndReorderBartDims(BartDimsEnum._10_TIME, BartDimsEnum._01_PHS1);
assertEquals(2, array2.ndim());
assertEquals(64, array2.shape(0));
assertEquals(128, array2.shape(1));
     * }</pre></blockquote>
     * 
     * @param selectedDims dimensions kept in the returned view
     * @return  a view that references this BartNDArray as parent, and
     * skips all singleton dimensions not included in the parameter list
     */
    public default BartNDArray selectAndReorderBartDims(BartDimsEnum... selectedDims) {
        // Select dimensions
        List<Integer> bartDims = getBartDimsAsIntList();
        List<Integer> integerSelectedDims = Stream.of(selectedDims)
            .mapToInt(BartDimsEnum::ordinal)
            .boxed()
            .collect(Collectors.toList());
        int[] intSelectedDims = integerSelectedDims.stream().mapToInt(bartDims::indexOf).toArray();
        BartNDArray selectView = selectedDims.length == ndim() ? this : selectDims(intSelectedDims);

        // Permute dimensions if needed
        List<Integer> selectedBartDims = selectView.getBartDimsAsIntList();
        int[] permutator = integerSelectedDims.stream().mapToInt(selectedBartDims::indexOf).toArray();
        return IntStream.range(0, selectedDims.length).allMatch(i -> i == permutator[i]) ?
            selectView : selectView.permuteDims(permutator);
    }

    @Override
    public default BartNDArray dropDims(int... droppedDims) {
        ViewOperations.checkDroppedDims(ndim(), droppedDims);
        List<Integer> set = ViewOperations.intArrayToList(droppedDims);
        if (set.size() == ndim()) throw new IllegalArgumentException(Errors.ALL_DIMS_DROPPED);
        return selectDims(IntStream.range(0, ndim()).filter(i -> !set.contains(i)).toArray());
    }

    @Override
    public default BartNDArray squeeze() {
        return selectDims(ViewOperations.getIndicesOfSingletonDims(shape()));
    }

    /**
     * Checks if meaning of dimensions are specified in BART or not.
     * 
     * @return true if the meaning of dimensions in BART are specified
     */
    public boolean areBartDimsSpecified();

    /**
     * Returns an array of BartDimsEnum that tells the meaning of each dimensions in BART.
     * 
     * @return an array of BartDimsEnum that tells the meaning of each dimensions in BART
     */
    public BartDimsEnum[] getBartDims();
    
    /**
     * Sets the meaning of dimensions.
     * 
     * @param bartDims meaning of dimensions in BART.
     */
    public void setBartDims(BartDimsEnum... bartDims);

    /**
     * Permutes and reshapes the array according to the meaning of dimensions in BART.
     * 
     * @return Prepared BartNDArray ready to pass to JNI backend
     */
    public default BartFloatNDArray prepareToPassToBackend() {
        List<Integer> intBartDims = getBartDimsAsIntList();
        int[] permutator = getBartDimsPermutator(intBartDims);
        BartNDArray view = permuteDims(permutator);
        int[] newShape = getBartDimsShape(intBartDims, permutator);
        BartNDArray result = view.reshape(newShape);
        return (BartFloatNDArray) (result instanceof BartFloatNDArray ? result : result.copy());
    }

    private List<Integer> getBartDimsAsIntList() {
        BartDimsEnum[] bartshape = getBartDims();
        return IntStream.range(0, bartshape.length)
            .mapToObj(i -> (Integer)bartshape[i].ordinal())
            .collect(Collectors.toList()); 
    }

    private int[] getBartDimsPermutator(List<Integer> bartDims) {
        List<Integer> sortedBartDims = bartDims.stream().sorted().collect(Collectors.toList());
        return IntStream.range(0, bartDims.size())
            .map(i -> bartDims.indexOf(sortedBartDims.get(i))).toArray();
    }
    
    private int[] getBartDimsShape(List<Integer> bartDims, int[] permutator) {
        Set<Integer> bartDimsSet = Set.copyOf(bartDims);
        int maxDim = Collections.max(bartDims);
        int[] counter = new int[1];
        return IntStream.range(0, maxDim + 1)
            .map(i -> bartDimsSet.contains(i) ? shape(permutator[counter[0]++]) : 1).toArray();
    }
    
}
