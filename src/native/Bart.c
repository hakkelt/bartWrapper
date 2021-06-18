#include <jni.h>        // JNI header provided by JDK
#include <stdio.h>      // C Standard IO Header
#include <string.h>
#include <malloc.h>
#include "io_github_hakkelt_bartwrapper_Bart.h"   // Generated

#include "main.h"
#include "misc/misc.h"
#include "misc/memcfl.h"
#include "misc/mri.h"
#include "misc/mmio.h"
#include "misc/io.h"

#define BUFFER_SIZE 4096
#define concat2(X, Y) X ## Y
#define concat(X, Y) concat2(X, Y)
#define jni_func(x) concat(Java_io_github_hakkelt_bartwrapper_Bart_, x)

static void run_bart_commands(char* output, JNIEnv *env, jobject thisObj, jobjectArray args)
{
    jclass thisClass = (*env)->GetObjectClass(env, thisObj);
    jfieldID fidSuccessFlag = (*env)->GetFieldID(env, thisClass, "successFlag", "Z");
    if (NULL == fidSuccessFlag) return;

    jsize dataLength = (*env)->GetArrayLength(env, args);
    char* commands[dataLength + 1];
    for (jsize i = 0; i < dataLength; i++) {
        jstring string = (jstring) ((*env)->GetObjectArrayElement(env, args, i));
        const char* str = (*env)->GetStringUTFChars(env, string, NULL);
        commands[i] = strdup(str);
        (*env)->ReleaseStringUTFChars(env, string, str);
    }
    commands[dataLength] = NULL;

    int ret = bart_command(output == NULL ? 0 : BUFFER_SIZE, output, dataLength, commands);
    for (jsize i = 0; i < dataLength; i++)
        free(commands[i]);

    if (ret == 0)
        (*env)->SetBooleanField(env, thisObj, fidSuccessFlag, JNI_TRUE);
}

JNIEXPORT void JNICALL jni_func(nativeRun)(JNIEnv *env, jobject thisObj, jobjectArray args)
{
    run_bart_commands(NULL, env, thisObj, args);
}

JNIEXPORT jstring JNICALL jni_func(nativeRead)(JNIEnv *env, jobject thisObj, jobjectArray args)
{
    char* output = (char*)malloc(BUFFER_SIZE);
    run_bart_commands(output, env, thisObj, args);
    jstring ret = (*env)->NewStringUTF(env, output);
    free(output);
    return ret;
}

JNIEXPORT void JNICALL jni_func(nativeRegisterMemory)(JNIEnv *env, jobject thisObj, jstring java_name, jintArray java_dims, jobject java_buffer)
{
    const char *name = (*env)->GetStringUTFChars(env, java_name, NULL);
    if (NULL == name) return;

    jint *dims = (*env)->GetIntArrayElements(env, java_dims, NULL);
    if (NULL == dims) return;
    jsize D = (*env)->GetArrayLength(env, java_dims);

    float complex* data = (float complex*)(*env)->GetDirectBufferAddress(env, java_buffer);
    if (NULL == data) return;

    jclass thisClass = (*env)->GetObjectClass(env, thisObj);
    jfieldID fidSuccessFlag = (*env)->GetFieldID(env, thisClass, "successFlag", "Z");
    if (NULL == fidSuccessFlag) return;

    if (memcfl_exists(name))
        memcfl_unlink(name);

    io_reserve_input(name);
    #ifdef _WIN32
        memcfl_register(name, D, dims, data, false);
    #else
        long longDims[D];
        for (jsize i = 0; i < D; i++)
            longDims[i] = dims[i];
        memcfl_register(name, D, longDims, data, false);
    #endif
    memcfl_unmap(data);
    
    (*env)->ReleaseIntArrayElements(env, java_dims, dims, 0);
    (*env)->ReleaseStringUTFChars(env, java_name, name);
    (*env)->SetBooleanField(env, thisObj, fidSuccessFlag, JNI_TRUE);
}

JNIEXPORT void JNICALL jni_func(nativeRegisterOutput)(JNIEnv *env, jobject thisObj, jstring java_name)
{
    const char *name = (*env)->GetStringUTFChars(env, java_name, NULL);
    if (NULL == name) return;

    jclass thisClass = (*env)->GetObjectClass(env, thisObj);
    jfieldID fidSuccessFlag = (*env)->GetFieldID(env, thisClass, "successFlag", "Z");
    if (NULL == fidSuccessFlag) return;

    io_reserve_output(name);
    
    (*env)->ReleaseStringUTFChars(env, java_name, name);
    (*env)->SetBooleanField(env, thisObj, fidSuccessFlag, JNI_TRUE);
}

JNIEXPORT jboolean JNICALL jni_func(nativeIsMemoryAssociated)(JNIEnv *env, jobject thisObj, jstring java_name)
{
    const char *name = (*env)->GetStringUTFChars(env, java_name, NULL);
    if (NULL == name) return false;

    jclass thisClass = (*env)->GetObjectClass(env, thisObj);
    jfieldID fidSuccessFlag = (*env)->GetFieldID(env, thisClass, "successFlag", "Z");
    if (NULL == fidSuccessFlag) return false;

    bool ret = memcfl_exists(name);
    
    (*env)->ReleaseStringUTFChars(env, java_name, name);
    (*env)->SetBooleanField(env, thisObj, fidSuccessFlag, JNI_TRUE);

    return ret;
}

JNIEXPORT jobject JNICALL jni_func(nativeLoadMemory)(JNIEnv *env, jobject thisObj, jstring java_name, jintArray java_dims)
{
    const char *name = (*env)->GetStringUTFChars(env, java_name, NULL);
    if (NULL == name) return NULL;

    jint *dims = (*env)->GetIntArrayElements(env, java_dims, NULL);
    if (NULL == dims) return NULL;
    jsize D = (*env)->GetArrayLength(env, java_dims);

    jclass thisClass = (*env)->GetObjectClass(env, thisObj);
    jfieldID fidSuccessFlag = (*env)->GetFieldID(env, thisClass, "successFlag", "Z");
    if (NULL == fidSuccessFlag) return NULL;

    io_reserve_input(name);
    #ifdef _WIN32
        void* data = (void*)load_cfl(name, D, dims);
        jint size = io_calc_size(D, dims, sizeof(complex float));
    #else
        long longDims[D];
        for (jsize i = 0; i < D; i++)
            longDims[i] = dims[i];
        void* data = (void*)load_cfl(name, D, longDims);
        jint size = io_calc_size(D, longDims, sizeof(complex float));
    #endif

    jobject buffer = (*env)->NewDirectByteBuffer(env, data, size);

    memcfl_unmap(data);

    (*env)->ReleaseIntArrayElements(env, java_dims, dims, 0);
    (*env)->ReleaseStringUTFChars(env, java_name, name);
    (*env)->SetBooleanField(env, thisObj, fidSuccessFlag, JNI_TRUE);

    return buffer;
}

JNIEXPORT void JNICALL jni_func(nativeUnregisterMemory)(JNIEnv *env, jobject thisObj, jstring java_name)
{
    const char *name = (*env)->GetStringUTFChars(env, java_name, NULL);
    if (NULL == name) return;

    jclass thisClass = (*env)->GetObjectClass(env, thisObj);
    jfieldID fidSuccessFlag = (*env)->GetFieldID(env, thisClass, "successFlag", "Z");
    if (NULL == fidSuccessFlag) return;

    memcfl_unlink(name);
    
    (*env)->ReleaseStringUTFChars(env, java_name, name);
    (*env)->SetBooleanField(env, thisObj, fidSuccessFlag, JNI_TRUE);
}
