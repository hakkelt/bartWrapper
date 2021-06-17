/* DO NOT EDIT THIS FILE - it is machine generated */
#include <jni.h>
/* Header for class io_github_hakkelt_bartconnector_BartConnector */

#ifndef _Included_io_github_hakkelt_bartconnector_BartConnector
#define _Included_io_github_hakkelt_bartconnector_BartConnector
#ifdef __cplusplus
extern "C" {
#endif
/*
 * Class:     io_github_hakkelt_bartconnector_BartConnector
 * Method:    nativeRun
 * Signature: ([Ljava/lang/String;)V
 */
JNIEXPORT void JNICALL Java_io_github_hakkelt_bartconnector_BartConnector_nativeRun
  (JNIEnv *, jobject, jobjectArray);

/*
 * Class:     io_github_hakkelt_bartconnector_BartConnector
 * Method:    nativeRead
 * Signature: ([Ljava/lang/String;)Ljava/lang/String;
 */
JNIEXPORT jstring JNICALL Java_io_github_hakkelt_bartconnector_BartConnector_nativeRead
  (JNIEnv *, jobject, jobjectArray);

/*
 * Class:     io_github_hakkelt_bartconnector_BartConnector
 * Method:    nativeRegisterMemory
 * Signature: (Ljava/lang/String;[ILjava/nio/FloatBuffer;)V
 */
JNIEXPORT void JNICALL Java_io_github_hakkelt_bartconnector_BartConnector_nativeRegisterMemory
  (JNIEnv *, jobject, jstring, jintArray, jobject);

/*
 * Class:     io_github_hakkelt_bartconnector_BartConnector
 * Method:    nativeIsNameRegistered
 * Signature: (Ljava/lang/String;)Z
 */
JNIEXPORT jboolean JNICALL Java_io_github_hakkelt_bartconnector_BartConnector_nativeIsNameRegistered
  (JNIEnv *, jobject, jstring);

/*
 * Class:     io_github_hakkelt_bartconnector_BartConnector
 * Method:    nativeRegisterOutput
 * Signature: (Ljava/lang/String;)V
 */
JNIEXPORT void JNICALL Java_io_github_hakkelt_bartconnector_BartConnector_nativeRegisterOutput
  (JNIEnv *, jobject, jstring);

/*
 * Class:     io_github_hakkelt_bartconnector_BartConnector
 * Method:    nativeLoadMemory
 * Signature: (Ljava/lang/String;[I)Ljava/nio/ByteBuffer;
 */
JNIEXPORT jobject JNICALL Java_io_github_hakkelt_bartconnector_BartConnector_nativeLoadMemory
  (JNIEnv *, jobject, jstring, jintArray);

/*
 * Class:     io_github_hakkelt_bartconnector_BartConnector
 * Method:    nativeUnregisterMemory
 * Signature: (Ljava/lang/String;)V
 */
JNIEXPORT void JNICALL Java_io_github_hakkelt_bartconnector_BartConnector_nativeUnregisterMemory
  (JNIEnv *, jobject, jstring);

#ifdef __cplusplus
}
#endif
#endif