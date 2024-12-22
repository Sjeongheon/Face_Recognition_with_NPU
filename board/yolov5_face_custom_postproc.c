/* ENLIGHT generated post_process.c calls these two functions after NPU inference.
 * Link with yolov5_face_postprocess.c and the target SDK's enlight_network API. */
#include "enlight_network.h"
#include "yolov5_face_postprocess.h"

static float p3[48 * 16 * 16];
static float p4[48 * 8 * 8];
static float p5[48 * 4 * 4];
static yolov5_face_detection_t detections[YOLOV5_FACE_MAX_CANDIDATES];
static int detection_count;

void custom_postproc_init(void)
{
    detection_count = 0;
}

int custom_postproc_run(int num_output, enlight_act_tensor_t **output_tensors)
{
    float *heads[3] = {p3, p4, p5};
    const float *readonly_heads[3] = {p3, p4, p5};
    int level, channel, y, x;
    if (num_output != 3 || !output_tensors) {
        return -1;
    }
    for (level = 0; level < 3; ++level) {
        const int side = 16 >> level;
        int dims[4];
        enlight_act_tensor_t *tensor = output_tensors[level];
        if (!tensor || tensor->data_type != ENLIGHT_DTYPE_INT8 || tensor->scale <= 0) {
            return -1;
        }
        enlight_get_tensor_dimensions(tensor, dims);
        if (dims[0] != 1 || dims[1] != 48 || dims[2] != side || dims[3] != side) {
            return -1;
        }
        for (channel = 0; channel < 48; ++channel) {
            for (y = 0; y < side; ++y) {
                for (x = 0; x < side; ++x) {
                    heads[level][(channel * side + y) * side + x] =
                        enlight_get_tensor_data_by_off(tensor, y, x, channel);
                }
            }
        }
    }
    detection_count = yolov5_face_decode_nms(readonly_heads, 0.4f, 0.5f,
                                               detections, YOLOV5_FACE_MAX_CANDIDATES);
    return detection_count;
}

/* Read these results after network inference. Coordinates use 128x128 letterbox space. */
const yolov5_face_detection_t *yolov5_face_get_detections(int *count)
{
    if (count) {
        *count = detection_count;
    }
    return detections;
}
