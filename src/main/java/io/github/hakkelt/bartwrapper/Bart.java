package io.github.hakkelt.bartwrapper;

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
import java.util.stream.IntStream;
import java.util.stream.Stream;

import io.github.hakkelt.ndarrays.NDArray;

import java.io.File;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;

/**
 * Driver for BART
 */
public class Bart {

    /**
     *
     */
    private static final String BART_DLL = "bart.dll";
    private static Set<String> alreadyUsedNames = new HashSet<>();
    private static Random random = new Random();
    private static boolean isInitialized = false;

    private static native void nativeRun(SuccessFlag successFlag, String ... args);
    private static native String nativeRead(SuccessFlag successFlag, String ... args);
    private static native void nativeregisterInput(SuccessFlag successFlag, String name, int[] dims, FloatBuffer buffer);
    private static native boolean nativeIsMemoryFileRegistered(SuccessFlag successFlag, String name);
    private static native void nativeRegisterOutput(SuccessFlag successFlag, String name);
    private static native ByteBuffer nativeLoadMemory(SuccessFlag successFlag, String name, int[] dims);
    private static native void nativeUnregisterInput(SuccessFlag successFlag, String name);

    class CannotCopyDllToTempException extends RuntimeException {
        public CannotCopyDllToTempException(String message) {
            super(message);
        }
    }

    static class SuccessFlag {
        private boolean flag = false;

        public void setValue() {
            flag = true;
        }

        public boolean getValue() {
            return flag;
        }
    }

    public static void init() {
        if (isInitialized) // if already initialized
            return;
        synchronized (Bart.class) {
            if (isInitialized) // if initialized while waiting for the lock
                return;
            new Bart();
            isInitialized = true;
        }
    }

    private Bart() {
        if (!loadDLLFromClassPath())
            loadDLLFromJarResources();
    }

    @FunctionalInterface
    interface ThrowingRunnable {
        public void run(SuccessFlag successFlag) throws BartException;
    }

    
    /** 
     * Register an NDArray to allow BART to use them as input.
     * 
     * <ul><li><b>Example:</b></li></ul>
     * 
     * <blockquote><pre>{@code 
BartNDArray array = new BartFloatNDArray(128, 128).fill(new Complex(1,-1));
Bart.registerInput("input.mem", array);
Bart.registerOutput("output.mem");
Bart.execute("cabs", "input.mem", "output.mem");
BartNDArray result = Bart.loadResult("output.mem"));
Bart.unregisterInput("input.mem");
Bart.unregisterInput("output.mem");
assertEquals(array.abs(), result.squeeze().real());
     * }</pre></blockquote>
     * 
     * @param name the registered NDArray can be referenced by this name
     * @param array NDArray to be used as input for BART commands
     * @throws BartException when JNI call fails for any reason
     */
    public static void registerInput(String name, NDArray<?> array) throws BartException {
        if (!name.endsWith(".mem"))
            throw new IllegalArgumentException(BartErrors.NAME_EXTENSION_IS_NOT_MEM);
        BartFloatNDArray arrayToPass;
        if (array instanceof BartNDArray) {
            if (((BartNDArray)array).areBartDimsSpecified())
                arrayToPass = ((BartNDArray)array).prepareToPassToBackend();
            else
                arrayToPass = (BartFloatNDArray)(array instanceof BartFloatNDArray ? array : ((BartNDArray)array).copy());
        } else {
            arrayToPass = new BartFloatNDArray(array);
        }
        int[] bartDims = arrayToPass.shape();
        FloatBuffer buffer = arrayToPass.getFloatBuffer();
        wrapJNIcall(successFlag -> nativeregisterInput(successFlag, name, bartDims, buffer));
    }

    /**
     * Check if a name is already registered for an input or an output.
     * 
     * @param name the name to be checked against
     * @return true if the name is already registered for an input or an output.
     * @throws BartException when JNI call fails for any reason
     */
    public static boolean isMemoryFileRegistered(String name) throws BartException {
        if (!name.endsWith(".mem"))
            throw new IllegalArgumentException(BartErrors.NAME_EXTENSION_IS_NOT_MEM);
        boolean[] ret = new boolean[1];
        wrapJNIcall(successFlag -> ret[0] = nativeIsMemoryFileRegistered(successFlag, name));
        return ret[0];
    }


    /** 
     * Register an name to be used as output of a later BART command.
     * 
     * <ul><li><b>Example:</b></li></ul>
     * 
     * <blockquote><pre>{@code 
BartNDArray array = new BartFloatNDArray(128, 128).fill(new Complex(1,-1));
Bart.registerInput("input.mem", array);
Bart.registerOutput("output.mem");
Bart.execute("cabs", "input.mem", "output.mem");
BartNDArray result = Bart.loadResult("output.mem"));
Bart.unregisterInput("input.mem");
Bart.unregisterInput("output.mem");
assertEquals(array.abs(), result.squeeze().real());
     * }</pre></blockquote>
     * 
     * @param name the registered NDArray can be referenced by this name
     * @throws BartException when JNI call fails for any reason
     */
    public static void registerOutput(String name) throws BartException {
        if (!name.endsWith(".mem"))
            throw new IllegalArgumentException(BartErrors.NAME_EXTENSION_IS_NOT_MEM);
        wrapJNIcall(successFlag -> nativeRegisterOutput(successFlag, name));
    }
    
    /** 
     * Load data from an in-memory BART file.
     * 
     * <ul><li><b>Example:</b></li></ul>
     * 
     * <blockquote><pre>{@code 
BartNDArray array = new BartFloatNDArray(128, 128).fill(new Complex(1,-1));
Bart.registerInput("input.mem", array);
Bart.registerOutput("output.mem");
Bart.execute("cabs", "input.mem", "output.mem");
BartNDArray result = Bart.loadResult("output.mem"));
Bart.unregisterInput("input.mem");
Bart.unregisterInput("output.mem");
     * }</pre></blockquote>
     * 
     * @param name the name of registered NDArray
     * @return NDArray
     * @throws BartException when JNI call fails for any reason
     */
    public static BartNDArray loadMemory(final String name) throws BartException {
        if (!name.endsWith(".mem"))
            throw new IllegalArgumentException(BartErrors.NAME_EXTENSION_IS_NOT_MEM);
        int[] dims = new int[16];
        ByteBuffer[] buffer = new ByteBuffer[1];
        wrapJNIcall(successFlag-> buffer[0] = nativeLoadMemory(successFlag, name, dims));
        buffer[0].order(ByteOrder.LITTLE_ENDIAN);
        BartNDArray array = new BartFloatNDArray(buffer[0], dims).copy();
        array.setBartDims(BartDimsEnum.values());
        return array;
    }

    /** 
     * Deletes a registered NDArray from BART's internal memory.
     * 
     * <ul><li><b>Example:</b></li></ul>
     * 
     * <blockquote><pre>{@code
BartNDArray array = new BartFloatNDArray(128, 128).fill(new Complex(1,-1));
Bart.registerInput("input.mem", array);
Bart.registerOutput("output.mem");
Bart.execute("cabs", "input.mem", "output.mem");
BartNDArray result = Bart.loadResult("output.mem"));
Bart.unregisterInput("input.mem");
Bart.unregisterInput("output.mem");
     * }</pre></blockquote>
     * 
     * @param name the name of the registered NDArray to be unregistered
     * @throws BartException when JNI call fails for any reason
     */
    public static void unregisterInput(String name) throws BartException {
        if (!name.endsWith(".mem"))
            throw new IllegalArgumentException(BartErrors.NAME_EXTENSION_IS_NOT_MEM);
        wrapJNIcall(successFlag -> nativeUnregisterInput(successFlag, name));
    }
    
    /** 
     * Execute a BART command
     * 
     * <ul><li><b>Example:</b></li></ul>
     * 
     * <blockquote><pre>{@code 
BartNDArray array = new BartFloatNDArray(128, 128).fill(new Complex(1,-1));
Bart.registerInput("input.mem", array);
Bart.registerOutput("output.mem");
Bart.execute("cabs", "input.mem", "output.mem");
BartNDArray result = Bart.loadResult("output.mem"));
Bart.unregisterInput("input.mem");
Bart.unregisterInput("output.mem");
     * }</pre></blockquote>
     * 
     * @param args name of BART command and its arguments
     * @throws BartException when JNI call fails for any reason
     */
    public static void execute(Object... args) throws BartException {
        List<String> registeredArrays = new ArrayList<>();
        wrapJNIcall(successFlag -> nativeRun(successFlag, convertInputs(registeredArrays, args)));
        for (String name : registeredArrays)
            unregisterInput(name);
    }
    
    /** 
     * Executes a BART command and reads its output to a String
     * 
     * <ul><li><b>Example:</b></li></ul>
     * 
     * <blockquote><pre>{@code 
String result = Bart.read("bitmask", "-b", 7);
     * }</pre></blockquote>
     * 
     * @param args name of BART command and its arguments
     * @return result of the BART command
     * @throws BartException when JNI call fails for any reason
     */
    public static String read(Object ... args) throws BartException {
        String[] ret = new String[1];
        List<String> registeredArrays = new ArrayList<>();
        String[] arguments = convertInputs(registeredArrays, args);
        wrapJNIcall(successFlag -> ret[0] = nativeRead(successFlag, arguments).split("\n")[0].trim());
        for (String name : registeredArrays)
            unregisterInput(name);
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
BartNDArray array = new BartFloatNDArray(128, 128).fill(new Complex(1,-1));
BartNDArray bartAbs = Bart.run("cabs", array).squeeze();
     * }</pre></blockquote>
     * 
     * @param args name of BART command and its arguments
     * @return NDArray that holds the output of the BART command
     * @throws BartException when JNI call fails for any reason
     */
    public static BartNDArray run(Object... args) throws BartException {
        List<String> registeredArrays = new ArrayList<>();
        String[] tmp = convertInputs(registeredArrays, args);
        String[] strArgs = new String[args.length + 1];
        IntStream.range(0, args.length).forEach(i -> strArgs[i] = tmp[i]);
        strArgs[args.length] = randomUniqueString(20) + ".mem";
        registerOutput(strArgs[args.length]);
        wrapJNIcall(successFlag -> nativeRun(successFlag, strArgs));
        BartNDArray result = loadMemory(strArgs[args.length]);
        unregisterInput(strArgs[args.length]);
        for (String name : registeredArrays)
            unregisterInput(name);
        return result;
    }

    protected boolean loadDLLFromJarResources() {
        File dllPath = new File(System.getenv("TMP") + File.separator + BART_DLL);
        dllPath.deleteOnExit();
        try {
            Files.copy(Bart.class.getResourceAsStream("/bart.dll"),
                dllPath.toPath(), StandardCopyOption.REPLACE_EXISTING);
            Logger.getLogger(Bart.class.getName())
                .log(Level.INFO, "JNI binary was copied to and loaded from the temp folder.");
        } catch (IOException e) {
            throw new CannotCopyDllToTempException(e.getMessage());
        }
        System.load(dllPath.getAbsolutePath());
        return true;
    }

    protected boolean loadDLLFromClassPath() {
        String[] classPath = System.getProperty("java.class.path").split(";");
        return Stream.of(classPath)
            .anyMatch(path -> {
                try {
                    if (path.matches(".*bartwrapper-\\d+\\.\\d+\\.\\d+\\.jar")) {
                        String dir = new File(path).getParentFile().getAbsolutePath();
                        System.load(dir + File.separator + BART_DLL);
                        return true;
                    } else if (new File(path).isDirectory()) {
                        System.load(path + File.separator + BART_DLL);
                        return true;
                    }
                    return false;
                } catch (UnsatisfiedLinkError e) {
                    return false;
                }
            });
    }
    
    protected static void wrapJNIcall(ThrowingRunnable func) throws BartException {
        init();
        SuccessFlag successFlag = new SuccessFlag();
        try {
            func.run(successFlag);
        } catch (Exception e) {
            throw new BartException(BartErrors.BART_FATAL);
        }
        if (!successFlag.getValue())
            throw new BartException(BartErrors.BART_UNSUCCESSFUL);
    }
    
    protected static String randomUniqueString(int length) {
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

    protected static String[] convertInputs(List<String> registeredArrays, Object... args) throws BartException {
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
                registerInput(name, (NDArray<?>)args[i]);
                strArgs[i] = name;
            } else
                throw new IllegalArgumentException(
                    String.format(BartErrors.INPUT_UNSUPPORTED_TYPE, args[i], args[i].getClass()));
        }
        return strArgs;
    }
 }