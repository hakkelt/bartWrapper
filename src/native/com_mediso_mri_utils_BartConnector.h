/* DO NOT EDIT THIS FILE - it is machine generated */
#include <jni.h>
/* Header for class com_mediso_mri_utils_BartConnector */

#ifndef _Included_com_mediso_mri_utils_BartConnector
#define _Included_com_mediso_mri_utils_BartConnector
#ifdef __cplusplus
extern "C" {
#endif
/*
 * Class:     com_mediso_mri_utils_BartConnector
 * Method:    nativeRun
 * Signature: ([Ljava/lang/String;)V
 */
JNIEXPORT void JNICALL Java_com_mediso_mri_utils_BartConnector_nativeRun
  (JNIEnv *, jobject, jobjectArray);

/*
 * Class:     com_mediso_mri_utils_BartConnector
 * Method:    nativeRead
 * Signature: ([Ljava/lang/String;)Ljava/lang/String;
 */
JNIEXPORT jstring JNICALL Java_com_mediso_mri_utils_BartConnector_nativeRead
  (JNIEnv *, jobject, jobjectArray);

/*
 * Class:     com_mediso_mri_utils_BartConnector
 * Method:    nativeRegisterMemory
 * Signature: (Ljava/lang/String;[ILjava/nio/FloatBuffer;)V
 */
JNIEXPORT void JNICALL Java_com_mediso_mri_utils_BartConnector_nativeRegisterMemory
  (JNIEnv *, jobject, jstring, jintArray, jobject);

/*
 * Class:     com_mediso_mri_utils_BartConnector
 * Method:    nativeLoadResult
 * Signature: (Ljava/lang/String;[I)Ljava/nio/ByteBuffer;
 */
JNIEXPORT jobject JNICALL Java_com_mediso_mri_utils_BartConnector_nativeLoadResult
  (JNIEnv *, jobject, jstring, jintArray);

#ifdef __cplusplus
}
#endif
#endif