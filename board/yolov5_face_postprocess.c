#include "yolov5_face_postprocess.h"

#include <math.h>
#include <stdlib.h>

static const float anchors[3][3][2] = {
    {{4, 5}, {8, 10}, {13, 16}},
    {{23, 29}, {43, 55}, {73, 105}},
    {{146, 217}, {231, 300}, {335, 433}},
};

typedef struct {
    yolov5_face_detection_t detection;
    size_t order;
} candidate_t;

static candidate_t candidates[YOLOV5_FACE_MAX_CANDIDATES];

static float logistic(float value)
{
    return 1.0f / (1.0f + expf(-value));
}

static float get_value(const float *head, int side, int channel, int y, int x)
{
    return head[(channel * side + y) * side + x];
}

static int compare_score_desc(const void *left, const void *right)
{
    const candidate_t *a = left;
    const candidate_t *b = right;
    const float score_a = a->detection.values[4];
    const float score_b = b->detection.values[4];
    if (score_a != score_b) {
        return (score_a < score_b) - (score_a > score_b);
    }
    return (a->order > b->order) - (a->order < b->order);
}

static float intersection_over_union(const float *a, const float *b)
{
    const float x1 = fmaxf(a[0], b[0]);
    const float y1 = fmaxf(a[1], b[1]);
    const float x2 = fminf(a[2], b[2]);
    const float y2 = fminf(a[3], b[3]);
    const float intersection = fmaxf(0, x2 - x1) * fmaxf(0, y2 - y1);
    const float a_area = fmaxf(0, a[2] - a[0]) * fmaxf(0, a[3] - a[1]);
    const float b_area = fmaxf(0, b[2] - b[0]) * fmaxf(0, b[3] - b[1]);
    const float union_area = a_area + b_area - intersection;
    return union_area > 0 ? intersection / union_area : 0;
}

int yolov5_face_decode_nms(const float *heads[YOLOV5_FACE_HEAD_COUNT],
                           float confidence_threshold, float iou_threshold,
                           yolov5_face_detection_t *results, size_t capacity)
{
    size_t count = 0;
    size_t kept = 0;
    int level, anchor, y, x, point;
    if (!heads || !heads[0] || !heads[1] || !heads[2] || !results ||
        capacity == 0 || confidence_threshold < 0 || confidence_threshold > 1 ||
        iou_threshold < 0 || iou_threshold > 1) {
        return -1;
    }

    for (level = 0; level < 3; ++level) {
        const int side = 16 >> level;
        const int stride = 8 << level;
        for (anchor = 0; anchor < 3; ++anchor) {
            const int channel = anchor * 16;
            const float anchor_w = anchors[level][anchor][0];
            const float anchor_h = anchors[level][anchor][1];
            for (y = 0; y < side; ++y) {
                for (x = 0; x < side; ++x) {
                    const float score = logistic(get_value(heads[level], side, channel + 4, y, x)) *
                                        logistic(get_value(heads[level], side, channel + 15, y, x));
                    float cx, cy, width, height;
                    float *row;
                    if (!(score > confidence_threshold)) {
                        continue;
                    }
                    candidates[count].order = count;
                    row = candidates[count++].detection.values;
                    cx = (logistic(get_value(heads[level], side, channel, y, x)) * 2 - 0.5f + x) * stride;
                    cy = (logistic(get_value(heads[level], side, channel + 1, y, x)) * 2 - 0.5f + y) * stride;
                    width = logistic(get_value(heads[level], side, channel + 2, y, x)) * 2;
                    height = logistic(get_value(heads[level], side, channel + 3, y, x)) * 2;
                    width = width * width * anchor_w;
                    height = height * height * anchor_h;
                    row[0] = cx - width / 2;
                    row[1] = cy - height / 2;
                    row[2] = cx + width / 2;
                    row[3] = cy + height / 2;
                    row[4] = score;
                    for (point = 0; point < 5; ++point) {
                        row[5 + point * 2] = get_value(heads[level], side, channel + 5 + point * 2, y, x)
                                                   * anchor_w + x * stride;
                        row[6 + point * 2] = get_value(heads[level], side, channel + 6 + point * 2, y, x)
                                                   * anchor_h + y * stride;
                    }
                }
            }
        }
    }

    qsort(candidates, count, sizeof(candidates[0]), compare_score_desc);
    for (size_t i = 0; i < count; ++i) {
        size_t j;
        for (j = 0; j < kept; ++j) {
            if (intersection_over_union(candidates[i].detection.values, results[j].values) > iou_threshold) {
                break;
            }
        }
        if (j == kept) {
            if (kept == capacity) {
                return -1;
            }
            results[kept++] = candidates[i].detection;
        }
    }
    return (int)kept;
}
