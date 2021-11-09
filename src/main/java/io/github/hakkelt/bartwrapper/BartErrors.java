package io.github.hakkelt.bartwrapper;

public class BartErrors {

    protected BartErrors() {}

    public static final String SET_BART_DIMS_SIZE_MISMATCH =
        "The length of the list of BART dimensions doesn't match the number of dimensions (%d)!";
    public static final String SET_BART_DIMS_DUPLICATES =
        "The list of BART dimensions contains duplicates!";
    public static final String UNINITIALIZED_BART_DIMS =
        "Meanings of dimension aren't specified yet!";
    public static final String CANNOT_SET_BART_DIMS_ON_VIEW =
        "Cannot set bartDims on a view!";
    public static final String BYTE_ORDER_IS_NOT_LITTLE_ENDIAN =
        "The byte order of the supplied ByteBuffer is big endian, but should be little endian!";
    public static final String BYTE_BUFFER_IS_NOT_DIRECT =
        "The the supplied ByteBuffer is not directly allocated!";

    public static final String BART_FATAL =
        "Fatal error occured while trying to run BART.";
    public static final String BART_UNSUCCESSFUL =
        "Running BART was unsuccessful.";
    public static final String INPUT_UNSUPPORTED_TYPE =
        "Cannot pass variable %s of type %s to BART!";
    public static final String NAME_EXTENSION_IS_NOT_MEM =
        "Name of array must end with '.mem'!";

}
