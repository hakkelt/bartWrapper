package io.github.hakkelt.bartwrapper;

/**
 * Exception that signals that something went wrong within the JNI call to a BART command.
 */
public class BartException extends Exception { 
    public BartException(String errorMessage) {
        super(errorMessage);
    }
}