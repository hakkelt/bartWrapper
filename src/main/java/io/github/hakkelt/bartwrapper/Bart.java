package io.github.hakkelt.bartwrapper;

import java.io.BufferedReader;
import java.io.File;
import java.io.IOException;
import java.io.InputStreamReader;
import java.nio.file.Files;
import java.nio.file.StandardCopyOption;
import java.util.ArrayList;
import java.util.List;
import java.util.function.Consumer;
import java.util.logging.Level;
import java.util.logging.Logger;
import java.util.stream.Collectors;
import java.util.stream.Stream;

import org.apache.commons.lang3.ArrayUtils;

import io.github.hakkelt.ndarrays.NDArray;

/**
 * Driver for BART
 */
public class Bart {

    private static final String TMPDIR = "java.io.tmpdir";
    private static final String BART_EXE = "bart.exe";
    private static final Logger LOGGER = Logger.getLogger(Bart.class.getName());
    private static File exePath;
    static {
        if (!searchExeInClassPath())
            copyExeFromJarResources();
    }

    private Bart() {}
    
    /** 
     * Execute a BART command
     * 
     * <ul><li><b>Example:</b></li></ul>
     * 
     * <blockquote><pre>{@code 
BartNDArray input = new BartFloatNDArray(128, 128).fill(new Complex(1,-1));
File output = File.createTempFile("bart_", ".ra");
Bart.execute("cabs", input, output);
BartNDArray result = BartNDArray.load(output));
     * }</pre></blockquote>
     * 
     * @param args name of BART command and its arguments
     * @throws BartException when running BART fails for any reason
     */
    public static void execute(Object... args) throws BartException {
        execute(null, args);
    }

    /** 
     * Execute a BART command
     * 
     * <ul><li><b>Example:</b></li></ul>
     * 
     * <blockquote><pre>{@code 
BartNDArray input = new BartFloatNDArray(128, 128).fill(new Complex(1,-1));
File output = File.createTempFile("bart_", ".ra");
StringBuilder str = new StringBuilder();
Bart.execute(outputLine -> str.append(outputLine).append(System.lineSeparator()),
    "cabs", input, output);
BartNDArray result = BartNDArray.load(output));
System.out.println(str.toString());
     * }</pre></blockquote>
     * 
     * @param outputConsumer a function that accepts a String as input. This function will receive
     * the output of the BART command line by line.
     * @param args name of BART command and its arguments
     * @throws BartException when running BART fails for any reason
     */
    public static void execute(Consumer<String> outputConsumer, Object... args) throws BartException {
        List<File> tempFiles = new ArrayList<>();
        try {
            Process process = new ProcessBuilder()
                .command(convertInputs(tempFiles, args))
                .directory(new File(System.getProperty(TMPDIR)))
                .start();
            handleProcessOutput(process, outputConsumer, false);
        } catch (IOException e) {
            throw new BartException(e.getMessage());
        } finally {
            cleanUp(tempFiles);
        }
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
     * @throws BartException when running BART fails for any reason
     */
    public static String read(Object ... args) throws BartException {
        List<File> tempFiles = new ArrayList<>();
        try {
            Process process = new ProcessBuilder()
                .command(convertInputs(tempFiles, args))
                .directory(new File(System.getProperty(TMPDIR)))
                .start();
            return handleProcessOutput(process, null, true).trim();
        } catch (IOException e) {
            throw new BartException(e.getMessage());
        } finally {
            cleanUp(tempFiles);
        }
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
     * @throws BartException when running Bart fails for any reason
     */
    public static BartNDArray run(Object... args) throws BartException {
        return run(null, args);
    }

    /** 
     * Executes a BART command and reads its output to an NDArray.
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
StringBuilder str = new StringBuilder();
BartNDArray bartAbs = Bart.run(outputLine -> str.append(outputLine).append(System.lineSeparator()),
    "cabs", array).squeeze();
System.out.println(str.toString());
     * }</pre></blockquote>
     * 
     * @param outputConsumer a function that accepts a String as input. This function will receive
     * the output of the BART command line by line.
     * @param args name of BART command and its arguments
     * @return NDArray that holds the output of the BART command
     * @throws BartException when running Bart fails for any reason
     */
    public static BartNDArray run(Consumer<String> outputConsumer, Object... args) throws BartException {
        List<File> tempFiles = new ArrayList<>();
        File output;
        try {
            output = Files.createTempFile("bart_", ".ra").toFile();
            Object[] args2 = ArrayUtils.add(args, output);
            Process process = new ProcessBuilder()
                .command(convertInputs(tempFiles, args2))
                .directory(new File(System.getProperty(TMPDIR)))
                .start();
            handleProcessOutput(process, outputConsumer, false);
            BartNDArray result = BartNDArray.load(output);
            result.setBartDims(Stream.of(BartDimsEnum.values()).limit(result.ndim()).toArray(BartDimsEnum[]::new));
            Files.delete(output.toPath());
            return result;
        } catch (IOException e) {
            throw new BartException(e.getMessage());
        } finally {
            cleanUp(tempFiles);
        }
    }

    protected static void copyExeFromJarResources() {
        exePath = new File(System.getenv("TMP") + File.separator + BART_EXE);
        if (exePath.exists())
            return;
        exePath.deleteOnExit();
        try {
            Files.copy(Bart.class.getResourceAsStream("/bart.exe"),
            exePath.toPath(), StandardCopyOption.REPLACE_EXISTING);
            LOGGER.log(Level.INFO, "BART executable was copied to the temp folder.");
        } catch (IOException e) {
            throw new ExceptionInInitializerError(e.getMessage());
        }
    }

    protected static boolean searchExeInClassPath() {
        return Stream.of(System.getProperty("java.class.path").split(";"))
            .anyMatch(path -> {
                File file = new File(path);
                if (file.isDirectory()) {
                    File[] containedFiles = file.listFiles((dir, name) -> name.equals(BART_EXE));
                    if (containedFiles.length > 0) {
                        exePath = containedFiles[0];
                        return true;
                    }
                } else if (file.getName().equals(BART_EXE)) {
                    exePath = file;
                    return true;
                }
                return false;
            });
    }

    protected static String[] convertInputs(List<File> tempFiles, Object... args) throws IOException {
        String[] strArgs = new String[args.length + 1];
        strArgs[0] = exePath.getAbsolutePath();
        for (int i = 0; i < args.length; i++) {
            if (args[i] instanceof String)
                strArgs[i + 1] = (String) args[i];
            else if (args[i] instanceof Float)
                strArgs[i + 1] = ((Float) args[i]).toString();
            else if (args[i] instanceof Double)
                strArgs[i + 1] = ((Double) args[i]).toString();
            else if (args[i] instanceof Integer)
                strArgs[i + 1] = ((Integer) args[i]).toString();
            else if (args[i] instanceof File)
                strArgs[i + 1] = ((File) args[i]).toString();
            else if (args[i] instanceof NDArray) {
                File file = BartNDArray.saveToTemp((NDArray<?>) args[i]);
                strArgs[i + 1] = file.getName();
                tempFiles.add(file);
            } else
                throw new IllegalArgumentException(
                    String.format(BartErrors.INPUT_UNSUPPORTED_TYPE, args[i], args[i].getClass()));
        }
        return strArgs;
    }

    private static void cleanUp(List<File> tempFiles) {
        for (File file : tempFiles) {
            try {
                Files.delete(file.toPath());
            } catch (IOException e) {
                LOGGER.warning("Could not delete " + file.getAbsolutePath());
            }
        }
    }

    private static String handleProcessOutput(Process process, Consumer<String> outputConsumer, boolean captureOutput) throws IOException, BartException {
        StringBuilder output = new StringBuilder();
        try (
            BufferedReader standard =
                new BufferedReader(new InputStreamReader(process.getInputStream()));        
            BufferedReader error = 
                new BufferedReader(new InputStreamReader(process.getErrorStream()))
        ) {
            String line = null;
            while ((line = standard.readLine()) != null) {  
                line = processString(line);
                if (captureOutput)
                    output.append(System.lineSeparator()).append(line);
                else if (outputConsumer != null)
                    outputConsumer.accept(line.trim());
                else
                    System.out.println(line); // NOSONAR
            }
            if (process.exitValue() != 0) {
                List<String> errors = error.lines()
                    .map(Bart::processString)
                    .collect(Collectors.toList());
                LOGGER.severe(() -> String.join(System.lineSeparator(), errors));
                String errorMessage = String.join(System.lineSeparator(), errors.stream()
                        //.filter(str -> str.startsWith("ERROR"))
                        .toArray(CharSequence[]::new));
                throw new BartException(String.format(BartErrors.BART_UNSUCCESSFUL, errorMessage.trim()));
            }
        }
        return output.toString();
    }

    private static String processString(String str) {
        return str.replaceAll("\u001B\\[[;\\d]*m", "").replace("\r", "");
    }
 }