#ifndef YOLOV5_FACE_POSTPROCESS_H
#define YOLOV5_FACE_POSTPROCESS_H

#include <stddef.h>

#define YOLOV5_FACE_HEAD_COUNT 3
#define YOLOV5_FACE_MAX_CANDIDATES 1008

/* Box (xyxy), confidence, then five (x, y) landmarks in 128x128 input space. */
typedef struct {
    float values[15];
} yolov5_face_detection_t;

/* Heads are dequantized, contiguous NCHW arrays of shape [1, 48, S, S],
 * where S is 16, 8, and 4 respectively. Returns -1 for invalid arguments.
 * Calls must be serialized because scratch storage is shared. */
int yolov5_face_decode_nms(const float *heads[YOLOV5_FACE_HEAD_COUNT],
                           float confidence_threshold, float iou_threshold,
                           yolov5_face_detection_t *results, size_t capacity);

/* Available when yolov5_face_custom_postproc.c is linked with the SDK. */
const yolov5_face_detection_t *yolov5_face_get_detections(int *count);

#endif
