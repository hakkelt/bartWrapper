package com.mediso.mri.utils;

import java.nio.FloatBuffer;
import java.util.HashSet;
import java.util.Random;
import java.util.Set;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

import org.apache.commons.math3.complex.Complex;

import java.io.File;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;

/**
 * Driver for BART
 */
public class BartConnector {
    static {
        File lib = new File("../../" + System.mapLibraryName("bart"));
        System.load(lib.getAbsolutePath());
    }
    static Set<String> alreadyUsedNames = new HashSet<>();
    private Random random = new Random();
    private boolean successFlag = false; // NOSONAR

    static final String ERROR_BART_FATAL = "Fatal error occured while trying to run BART.";
    static final String ERROR_BART_UNSUCCESSFUL = "Running BART was unsuccessful.";
    static final String ERROR_INPUT_UNSUPPORTED_TYPE = "Cannot pass variable %s of type %s to BART!";

    private native void nativeRun(String ... args);
    private native String nativeRead(String ... args);
    private native void nativeRegisterMemory(String name, int[] dims, FloatBuffer buffer);
    private native ByteBuffer nativeLoadResult(String name, int[] dims);

    @FunctionalInterface
    interface ThrowingRunnable {
        public void run() throws BartException;
    }

    
    protected void wrapJNIcall(ThrowingRunnable func) throws BartException {
        try {
            successFlag = false;
            func.run();
        } catch (Exception e) {
            throw new BartException(ERROR_BART_FATAL);
        }
        if (!successFlag) //NOSONAR
            throw new BartException(ERROR_BART_UNSUCCESSFUL);
    }

    
    protected String randomString(int length) {
        int leftLimit = 97; // letter 'a'
        int rightLimit = 122; // letter 'z'
        String str;
        do {
            str = random.ints(leftLimit, rightLimit + 1)
                .limit(length)
                .collect(StringBuilder::new, StringBuilder::appendCodePoint, StringBuilder::append)
                .toString();
        } while (alreadyUsedNames.contains(str));
        alreadyUsedNames.add(str);
        return str;
    }

    
    protected static int[] getBartDimsPermutator(int[] dimsOrder, int ndims) {
        int[] sorted = IntStream.of(dimsOrder).sorted().toArray();
        int[] permutator = new int[ndims];
        for (int i = 0; i < ndims; i++)
            for (int j = 0; j < ndims; j++)
                if (dimsOrder[j] == sorted[i]) {
                    permutator[i] = j;
                    break;
                }
        return permutator;
    }

    
    protected int[] getBartDimsAsIntArray(NDArray<?> array) {
        BartDimsEnum[] bartdims = array.getBartDims();
        return IntStream.range(0, bartdims.length)
            .map(i -> bartdims[i].ordinal()).toArray();
    }


    protected NDArray<?> permuteByBartDims(NDArray<?> array) { //NOSONAR
        int[] bartDims = getBartDimsAsIntArray(array);
        int[] sortedBartDims = IntStream.of(bartDims).sorted().toArray();
        int[] permutator = IntStream.range(0, bartDims.length)
            .map(i ->
                IntStream.range(0, bartDims.length)
                    .filter(j -> sortedBartDims[i] == bartDims[j])
                    .findFirst().getAsInt()
            ).toArray();
        return array.permuteDims(permutator);
    }

    
    protected NDArray<?> reshapeByBartDims(NDArray<?> array) { //NOSONAR
        int[] bartDims = getBartDimsAsIntArray(array);
        Set<Integer> bartDimsSet = IntStream.of(bartDims).boxed().collect(Collectors.toSet());
        int maxDim = IntStream.of(bartDims).max().getAsInt();
        int[] counter = new int[1];
        int[] newShape = IntStream.range(0, maxDim + 1).map(i -> bartDimsSet.contains(i) ? array.dims(counter[0]++) : 1).toArray();
        return array.reshape(newShape);
    }

    
    /** 
     * Register an NDArray to allow BART to use them as input.
     * 
     * <p>Example code:<pre>{@code 
BartConnector bart = new BartConnector();
bart.registerMemory("input.mem", array);
bart.execute("cabs", "input.mem", "output.mem");
NDArray<Complex> result = bart.loadResult("output.mem"));
assertEquals(array.abs(), result.squeeze().real());
     * }</pre>
     * 
     * @param name The registered NDArray can be referenced by this name
     * @param array NDArray to be used as input for BART commands
     * @throws BartException
     */
    public void registerMemory(String name, NDArray<?> array) throws BartException {
        if (!name.endsWith(".mem"))
            throw new IllegalArgumentException("Name of array must end with '.mem'!");
        if (array.areBartDimsSpecified()) {
            array = permuteByBartDims(array);
            array = reshapeByBartDims(array);
        }
        NDArray<?> arrayToPass = array.dataTypeAsString().equals("ComplexF32") ?
                array :
                new ComplexF32NDArray(array);
        FloatBuffer buffer = (FloatBuffer)arrayToPass.getBuffer();
        wrapJNIcall(() -> nativeRegisterMemory(name, arrayToPass.dims(), buffer));
    }

    
    /** 
     * Load result from an in-memory BART file.
     * 
     * <p>Example code:<pre>{@code 
BartConnector bart = new BartConnector();
bart.registerMemory("input.mem", array);
bart.execute("cabs", "input.mem", "output.mem");
NDArray<Complex> result = bart.loadResult("output.mem"));
assertEquals(array.abs(), result.squeeze().real());
     * }</pre>
     * 
     * @param name
     * @return NDArray<Complex>
     * @throws BartException
     */
    public NDArray<Complex> loadResult(final String name) throws BartException {
        if (!name.endsWith(".mem"))
            throw new IllegalArgumentException("Name of array must end with '.mem'!");
        int[] dims = new int[16];
        ByteBuffer[] buffer = new ByteBuffer[1];
        wrapJNIcall(()-> buffer[0] = this.nativeLoadResult(name, dims));
        buffer[0].order(ByteOrder.LITTLE_ENDIAN);
        NDArray<Complex> array = new ComplexF32NDArray(dims, buffer[0].asFloatBuffer());
        array.setBartDims(BartDimsEnum.values());
        return array;
    }
    

    protected String[] convertInputs(Object... args) throws BartException {
        String[] strArgs = new String[args.length];
        for (int i = 0; i < args.length; i++) {
            if (args[i] instanceof String)
                strArgs[i] = (String)args[i];
            else if (args[i] instanceof Float)
                strArgs[i] = ((Float)args[i]).toString();
            else if (args[i] instanceof Double)
                strArgs[i] = ((Double)args[i]).toString();
            else if (args[i] instanceof Integer)
                strArgs[i] = ((Integer)args[i]).toString();
            else if (args[i] instanceof NDArray) {
                String name = randomString(20) + ".mem";
                registerMemory(name, (NDArray<?>)args[i]);
                strArgs[i] = name;
            } else
                throw new IllegalArgumentException(
                    String.format(ERROR_INPUT_UNSUPPORTED_TYPE, args[i], args[i].getClass()));
        }
        return strArgs;
    }

    
    /** 
     * Execute a BART command
     * 
     * <p>Example code:<pre>{@code 
NDArray<Complex> array = new ComplexF32NDArray(128, 128).fill(new Complex(1,-1));
BartConnector bart = new BartConnector();
bart.registerMemory("input.mem", array);
bart.execute("cabs", "input.mem", "output.mem");
NDArray<Complex> result = bart.loadResult("output.mem"));
assertEquals(array.abs(), result.squeeze().real());
     * }</pre>
     * 
     * @param args
     * @throws BartException
     */
    public void execute(Object... args) throws BartException {
        wrapJNIcall(() -> nativeRun(convertInputs(args)));
    }
    
    
    /** 
     * Executes a BART command and reads its output to a String
     * 
     * <p>Example code:<pre>{@code 
BartConnector bart = new BartConnector();
assertEquals("0 1 2", bart.read("bitmask", "-b", 7));
     * }</pre>
     * 
     * @param args Name of BART command and its arguments
     * @return result of the BART command
     * @throws BartException
     */
    public String read(Object ... args) throws BartException {
        String[] ret = new String[1];
        wrapJNIcall(() -> ret[0] = nativeRead(convertInputs(args)).split("\n")[0].trim());
        return ret[0];
    }

    
    /** 
     * Executes a BART command and reads its output to an NDArray
     * 
     * <p>Note: It is assumed that the BART command to be executed
     * saves its output to a file and it expects the name of the file
     * to be specified as the last argument. When passing arguments to this 
     * function, this last argument specifying the output file name should be 
     * omitted as it is handled automaticall by this driver.
     * 
     * <p>Example code:<pre>{@code 
NDArray<Complex> array = new ComplexF32NDArray(128, 128).fill(new Complex(1,-1));
NDArray<Double> javaAbs = array.abs();
NDArray<Complex> bartAbs = bart.run("cabs", array).squeeze();
assertTrue(bartAbs.subtract(javaAbs).abs().sum() / array.length() < 1e-7);
     * }</pre>
     * 
     * @param args
     * @return NDArray<Complex>
     * @throws BartException
     */
    public NDArray<Complex> run(Object... args) throws BartException {
        String[] tmp = convertInputs(args);
        String[] strArgs = new String[args.length + 1];
        IntStream.range(0, args.length).forEach(i -> strArgs[i] = tmp[i]);
        strArgs[args.length] = randomString(20) + ".mem";
        wrapJNIcall(() -> nativeRun(strArgs));
        return loadResult(strArgs[args.length]);
    }
 }