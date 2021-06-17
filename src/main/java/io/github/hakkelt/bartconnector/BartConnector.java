package io.github.hakkelt.bartconnector;

import java.nio.FloatBuffer;
import java.nio.file.Files;
import java.nio.file.StandardCopyOption;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Random;
import java.util.Set;
import java.util.logging.Level;
import java.util.logging.Logger;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

import org.apache.commons.math3.complex.Complex;

import io.github.hakkelt.ndarrays.NDArray;
import io.github.hakkelt.ndarrays.ComplexF32NDArray;
import io.github.hakkelt.ndarrays.BartDimsEnum;

import java.io.File;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;

/**
 * Driver for BART
 */
public class BartConnector {
    static Set<String> alreadyUsedNames = new HashSet<>();
    private Random random = new Random();
    private boolean successFlag = false; // NOSONAR

    static final String ERROR_BART_FATAL = "Fatal error occured while trying to run BART.";
    static final String ERROR_BART_UNSUCCESSFUL = "Running BART was unsuccessful.";
    static final String ERROR_INPUT_UNSUPPORTED_TYPE = "Cannot pass variable %s of type %s to BART!";
    static final String ERROR_NAME_EXTENSION_IS_NOT_MEM = "Name of array must end with '.mem'!";

    private native void nativeRun(String ... args);
    private native String nativeRead(String ... args);
    private native void nativeRegisterMemory(String name, int[] dims, FloatBuffer buffer);
    private native boolean nativeIsNameRegistered(String name);
    private native void nativeRegisterOutput(String name);
    private native ByteBuffer nativeLoadMemory(String name, int[] dims);
    private native void nativeUnregisterMemory(String name);

    private static BartConnector singleInstance = null;

    private BartConnector() throws IOException {
        final Logger logger = Logger.getLogger(BartConnector.class.getName());

        if (!this.getClass().getResource("").getProtocol().equals("jar")) {
            // During development...
            String workingDirectory = System.getProperty("user.dir").replace("\\", "/");
            String location = (workingDirectory.endsWith("/src/test") ?
                workingDirectory.substring(0, workingDirectory.length() - "/src/test".length()) :
                workingDirectory) + "/src/main/resources";
            String libName = System.mapLibraryName("bart");
            String libPath = String.format("%s/%s", location, libName);
            System.load(libPath);
        } else {
            try {
                System.loadLibrary("bart");
            } catch (UnsatisfiedLinkError e2) {
                loadDLLFromJarResources();
                logger.log(Level.INFO, "JNI binary was copied to and loaded from the temp folder.");
            }
        }

    }

    /**
     * Retrieves a singleton instance of BART driver.
     * 
     * The first call to this method copies the dll to the temp directory,
     * and loads it, and returns with the driver call. The subsequent calls, however,
     * only returns the already created driver class.
     * 
     * @return singleton instance of BART driver
     * @throws IOException when an error occurs during copying the dll to the temp directory
     */
    public static BartConnector getInstance() throws IOException {
        if (singleInstance == null)
            singleInstance = new BartConnector();
        return singleInstance;
    }

    @FunctionalInterface
    interface ThrowingRunnable {
        public void run() throws BartException;
    }

    
    /** 
     * Register an NDArray to allow BART to use them as input.
     * 
     * <ul><li><b>Example:</b></li></ul>
     * 
     * <blockquote><pre>{@code 
NDArray<Complex> array = new ComplexF32NDArray(128, 128).fill(new Complex(1,-1));
BartConnector bart = BartConnector.getInstance();
bart.registerMemory("input.mem", array);
bart.registerOutput("output.mem");
bart.execute("cabs", "input.mem", "output.mem");
NDArray<Complex> result = bart.loadResult("output.mem"));
bart.unregisterMemory("input.mem");
bart.unregisterMemory("output.mem");
assertEquals(array.abs(), result.squeeze().real());
     * }</pre></blockquote>
     * 
     * @param name the registered NDArray can be referenced by this name
     * @param array NDArray to be used as input for BART commands
     * @throws BartException when JNI call fails for any reason
     */
    public void registerMemory(String name, NDArray<?> array) throws BartException {
        if (!name.endsWith(".mem"))
            throw new IllegalArgumentException(ERROR_NAME_EXTENSION_IS_NOT_MEM);
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
     * Check if a name is already registered for an input or an output.
     * 
     * @param name the name to be checked against
     * @return true if the name is already registered for an input or an output.
     * @throws BartException when JNI call fails for any reason
     */
    public boolean isNameRegistered(String name) throws BartException {
        if (!name.endsWith(".mem"))
            throw new IllegalArgumentException(ERROR_NAME_EXTENSION_IS_NOT_MEM);
        boolean[] ret = new boolean[1];
        wrapJNIcall(() -> ret[0] = nativeIsNameRegistered(name));
        return ret[0];
    }


    /** 
     * Register an name to be used as output of a later BART command.
     * 
     * <ul><li><b>Example:</b></li></ul>
     * 
     * <blockquote><pre>{@code 
NDArray<Complex> array = new ComplexF32NDArray(128, 128).fill(new Complex(1,-1));
BartConnector bart = BartConnector.getInstance();
bart.registerMemory("input.mem", array);
bart.registerOutput("output.mem");
bart.execute("cabs", "input.mem", "output.mem");
NDArray<Complex> result = bart.loadResult("output.mem"));
bart.unregisterMemory("input.mem");
bart.unregisterMemory("output.mem");
assertEquals(array.abs(), result.squeeze().real());
     * }</pre></blockquote>
     * 
     * @param name the registered NDArray can be referenced by this name
     * @throws BartException when JNI call fails for any reason
     */
    public void registerOutput(String name) throws BartException {
        wrapJNIcall(() -> nativeRegisterOutput(name));
    }
    
    /** 
     * Load data from an in-memory BART file.
     * 
     * <ul><li><b>Example:</b></li></ul>
     * 
     * <blockquote><pre>{@code 
NDArray<Complex> array = new ComplexF32NDArray(128, 128).fill(new Complex(1,-1));
BartConnector bart = BartConnector.getInstance();
bart.registerMemory("input.mem", array);
bart.registerOutput("output.mem");
bart.execute("cabs", "input.mem", "output.mem");
NDArray<Complex> result = bart.loadResult("output.mem"));
bart.unregisterMemory("input.mem");
bart.unregisterMemory("output.mem");
     * }</pre></blockquote>
     * 
     * @param name the name of registered NDArray
     * @return NDArray
     * @throws BartException when JNI call fails for any reason
     */
    public NDArray<Complex> loadMemory(final String name) throws BartException {
        if (!name.endsWith(".mem"))
            throw new IllegalArgumentException(ERROR_NAME_EXTENSION_IS_NOT_MEM);
        int[] dims = new int[16];
        ByteBuffer[] buffer = new ByteBuffer[1];
        wrapJNIcall(()-> buffer[0] = this.nativeLoadMemory(name, dims));
        buffer[0].order(ByteOrder.LITTLE_ENDIAN);
        NDArray<Complex> array = new ComplexF32NDArray(dims, buffer[0].asFloatBuffer()).copy();
        array.setBartDims(BartDimsEnum.values());
        return array;
    }

    /** 
     * Deletes a registered NDArray from BART's internal memory.
     * 
     * <ul><li><b>Example:</b></li></ul>
     * 
     * <blockquote><pre>{@code
NDArray<Complex> array = new ComplexF32NDArray(128, 128).fill(new Complex(1,-1));
BartConnector bart = BartConnector.getInstance();
bart.registerMemory("input.mem", array);
bart.registerOutput("output.mem");
bart.execute("cabs", "input.mem", "output.mem");
NDArray<Complex> result = bart.loadResult("output.mem"));
bart.unregisterMemory("input.mem");
bart.unregisterMemory("output.mem");
     * }</pre></blockquote>
     * 
     * @param name the name of the registered NDArray to be unregistered
     * @throws BartException when JNI call fails for any reason
     */
    public void unregisterMemory(String name) throws BartException {
        if (!name.endsWith(".mem"))
            throw new IllegalArgumentException(ERROR_NAME_EXTENSION_IS_NOT_MEM);
        wrapJNIcall(() -> nativeUnregisterMemory(name));
    }

    
    /** 
     * Execute a BART command
     * 
     * <ul><li><b>Example:</b></li></ul>
     * 
     * <blockquote><pre>{@code 
NDArray<Complex> array = new ComplexF32NDArray(128, 128).fill(new Complex(1,-1));
BartConnector bart = BartConnector.getInstance();
bart.registerMemory("input.mem", array);
bart.registerOutput("output.mem");
bart.execute("cabs", "input.mem", "output.mem");
NDArray<Complex> result = bart.loadResult("output.mem"));
bart.unregisterMemory("input.mem");
bart.unregisterMemory("output.mem");
     * }</pre></blockquote>
     * 
     * @param args name of BART command and its arguments
     * @throws BartException when JNI call fails for any reason
     */
    public void execute(Object... args) throws BartException {
        List<String> registeredArrays = new ArrayList<>();
        wrapJNIcall(() -> nativeRun(convertInputs(registeredArrays, args)));
        for (String name : registeredArrays)
            unregisterMemory(name);
    }
    
    
    /** 
     * Executes a BART command and reads its output to a String
     * 
     * <ul><li><b>Example:</b></li></ul>
     * 
     * <blockquote><pre>{@code 
BartConnector bart = BartConnector.getInstance();
String result = bart.read("bitmask", "-b", 7);
     * }</pre></blockquote>
     * 
     * @param args name of BART command and its arguments
     * @return result of the BART command
     * @throws BartException when JNI call fails for any reason
     */
    public String read(Object ... args) throws BartException {
        String[] ret = new String[1];
        List<String> registeredArrays = new ArrayList<>();
        wrapJNIcall(() -> ret[0] = nativeRead(convertInputs(registeredArrays, args)).split("\n")[0].trim());
        for (String name : registeredArrays)
            unregisterMemory(name);
        return ret[0];
    }

    
    /** 
     * Executes a BART command and reads its output to an NDArray
     * 
     * <p>Note: It is assumed that the BART command to be executed
     * saves its output to a file and it expects the name of the file
     * to be specified as the last argument. When passing arguments to this 
     * function, this last argument specifying the output file name should be 
     * omitted as it is handled automaticall by this driver.</p>
     * 
     * <ul><li><b>Example:</b></li></ul>
     * 
     * <blockquote><pre>{@code 
NDArray<Complex> array = new ComplexF32NDArray(128, 128).fill(new Complex(1,-1));
BartConnector bart = BartConnector.getInstance();
NDArray<Complex> bartAbs = bart.run("cabs", array).squeeze();
     * }</pre></blockquote>
     * 
     * @param args name of BART command and its arguments
     * @return NDArray that holds the output of the BART command
     * @throws BartException when JNI call fails for any reason
     */
    public NDArray<Complex> run(Object... args) throws BartException {
        List<String> registeredArrays = new ArrayList<>();
        String[] tmp = convertInputs(registeredArrays, args);
        String[] strArgs = new String[args.length + 1];
        IntStream.range(0, args.length).forEach(i -> strArgs[i] = tmp[i]);
        strArgs[args.length] = randomUniqueString(20) + ".mem";
        wrapJNIcall(() -> registerOutput(strArgs[args.length]));
        wrapJNIcall(() -> nativeRun(strArgs));
        NDArray<Complex> result = loadMemory(strArgs[args.length]);
        unregisterMemory(strArgs[args.length]);
        for (String name : registeredArrays)
            unregisterMemory(name);
        return result;
    }

    protected void loadDLLFromJarResources() throws IOException {
        File dllPath = new File(System.getenv("TMP") + File.separator + "bart.dll");
        dllPath.deleteOnExit();

        Files.copy(BartConnector.class.getResourceAsStream("/bart.dll"),
            dllPath.toPath(), StandardCopyOption.REPLACE_EXISTING);

        System.load(dllPath.getAbsolutePath());
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

    
    protected String randomUniqueString(int length) {
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


    protected String[] convertInputs(List<String> registeredArrays, Object... args) throws BartException {
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
                String name = randomUniqueString(20) + ".mem";
                registeredArrays.add(name);
                registerMemory(name, (NDArray<?>)args[i]);
                strArgs[i] = name;
            } else
                throw new IllegalArgumentException(
                    String.format(ERROR_INPUT_UNSUPPORTED_TYPE, args[i], args[i].getClass()));
        }
        return strArgs;
    }
 }