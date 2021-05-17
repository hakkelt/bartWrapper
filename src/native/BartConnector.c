#include <jni.h>        // JNI header provided by JDK
#include <stdio.h>      // C Standard IO Header
#include <string.h>
#include <malloc.h>
#include "com_mediso_mri_utils_BartConnector.h"   // Generated

#include "main.h"
#include "misc/misc.h"
#include "misc/memcfl.h"
#include "misc/mri.h"
#include "misc/mmio.h"

#define BUFFER_SIZE 4096

static void run_bart_commands(char* output, JNIEnv *env, jobject thisObj, jobjectArray args)
{
    jclass thisClass = (*env)->GetObjectClass(env, thisObj);
    jfieldID fidSuccessFlag = (*env)->GetFieldID(env, thisClass, "successFlag", "Z");
    if (NULL == fidSuccessFlag) return;

    jsize dataLength = (*env)->GetArrayLength(env, args);
    char** commands = (char**)calloc(dataLength + 1, sizeof(char*));
    for (jsize i = 0; i < dataLength; i++) {
        jstring string = (jstring) ((*env)->GetObjectArrayElement(env, args, i));
        commands[i] = strdup((*env)->GetStringUTFChars(env, string, NULL));
    }
    commands[dataLength] = NULL;

    int ret = bart_command(output == NULL ? 0 : BUFFER_SIZE, output, dataLength, commands);
    for (jsize i = 0; i < dataLength; i++)
        free(commands[i]);
    free(commands);

    if (ret == 0)
        (*env)->SetBooleanField(env, thisObj, fidSuccessFlag, JNI_TRUE);
}

JNIEXPORT void JNICALL Java_com_mediso_mri_utils_BartConnector_nativeRun(JNIEnv *env, jobject thisObj, jobjectArray args)
{
    run_bart_commands(NULL, env, thisObj, args);
}

JNIEXPORT jstring JNICALL Java_com_mediso_mri_utils_BartConnector_nativeRead(JNIEnv *env, jobject thisObj, jobjectArray args)
{
    char* output = (char*)malloc(BUFFER_SIZE);
    run_bart_commands(output, env, thisObj, args);
    jstring ret = (*env)->NewStringUTF(env, output);
    free(output);
    return ret;
}

JNIEXPORT void JNICALL Java_com_mediso_mri_utils_BartConnector_nativeRegisterMemory(JNIEnv *env, jobject thisObj, jstring java_name, jintArray java_dims, jobject java_buffer)
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

    memcfl_register(name, D, dims, data, false);
    
    (*env)->SetBooleanField(env, thisObj, fidSuccessFlag, JNI_TRUE);
    (*env)->ReleaseIntArrayElements(env, java_dims, dims, 0);
    (*env)->ReleaseStringUTFChars(env, java_name, name);
}

JNIEXPORT jobject JNICALL Java_com_mediso_mri_utils_BartConnector_nativeLoadResult(JNIEnv *env, jobject thisObj, jstring java_name, jintArray java_dims)
{
    const char *name = (*env)->GetStringUTFChars(env, java_name, NULL);
    if (NULL == name) return NULL;

    jint *dims = (*env)->GetIntArrayElements(env, java_dims, NULL);
    if (NULL == dims) return NULL;
    jsize D = (*env)->GetArrayLength(env, java_dims);

    jclass thisClass = (*env)->GetObjectClass(env, thisObj);
    jfieldID fidSuccessFlag = (*env)->GetFieldID(env, thisClass, "successFlag", "Z");
    if (NULL == fidSuccessFlag) return NULL;

    void* data = (void*)load_cfl(name, D, dims);
    jlong size = io_calc_size(D, dims, sizeof(complex float));

    jobject buffer = (*env)->NewDirectByteBuffer(env, data, size);
    (*env)->ReleaseIntArrayElements(env, java_dims, dims, 0);
    
    (*env)->SetBooleanField(env, thisObj, fidSuccessFlag, JNI_TRUE);

    return buffer;
}
