// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

#include "inference.hpp"
#include <immintrin.h>
#include <iostream>
#include <cstdint>
#include <string>
#include <vector>
#include <cmath>
#include <algorithm>
#include <cstring>
#include <stdexcept>

#define IOU_THRESHOLD 0.45

uint16_t float32_to_float16(float value) {
    __m128 input = _mm_set_ss(value);
    return _mm_cvtsi128_si32(_mm_cvtps_ph(input, 0));
}

float float16_to_float32(uint16_t half) {
    uint16_t sign = (half & 0x8000) >> 15;
    uint16_t exponent = (half & 0x7C00) >> 10;
    uint16_t mantissa = (half & 0x03FF);

    if (exponent == 0) {
        if (mantissa == 0) {
            return sign ? -0.0f : 0.0f;
        }
        return std::ldexp(mantissa / 1024.0f, -14) * (sign ? -1.0f : 1.0f);
    }
    else if (exponent == 31) {
        return mantissa ? NAN : (sign ? -INFINITY : INFINITY);
    }

    float real_value = std::ldexp(1.0f + mantissa / 1024.0f, exponent - 15);
    return sign ? -real_value : real_value;
}


void Image::preprocess(cv::Mat* img, std::vector<uint16_t>& triton_data, int input_w, int input_h)
{
    int w, h, x, y;
    float r_w = input_w / (img->cols*1.0);
    float r_h = input_h / (img->rows*1.0);
    if (r_h > r_w) {
        w = input_w;
        h = r_w * img->rows;
        x = 0;
        y = (input_h - h) / 2;
    } else {
        w = r_h * img->cols;
        h = input_h;
        x = (input_w - w) / 2;
        y = 0;
    }
    cv::Mat re(h, w, CV_8UC3);
    cv::resize(*img, re, re.size(), 0, 0, cv::INTER_LINEAR);
    cv::Mat out(input_h, input_w, CV_8UC3, cv::Scalar(114, 114, 114));
    re.copyTo(out(cv::Rect(x, y, re.cols, re.rows)));
    cv::cvtColor(out, out, cv::COLOR_BGR2RGB);
    unsigned char* data = (unsigned char*)out.data;
    int step = out.step;

    for (int yy = 0; yy < input_h; ++yy)
    {
        for (int kk = 0; kk < 3; ++kk)
        {
            for (int xx = 0; xx < input_w; ++xx)
            {
                float temp_f = data[yy * step + xx * 3 + kk] / 255.0f;

                triton_data[kk * input_w * input_h + yy * input_w + xx] = float32_to_float16(temp_f);

            }
        }
    }
}

int getDetectionsFromTritonRawData(std::vector<float>& raw, std::vector<detection_struct>& detections,
    std::vector<std::string>& classes, float threshold, int image_width, int image_height)
{
    detections.clear();
    const size_t channels = classes.size() + 4;
    if (classes.empty() || image_width <= 0 || image_height <= 0 || raw.size() % channels != 0)
        throw std::invalid_argument("Invalid detection tensor or image dimensions");
    const size_t anchors = raw.size() / channels;
    std::vector<BoundingBox> boxes;
    for (size_t i = 0; i < anchors; ++i) {
        size_t best = 0;
        float score = raw[4 * anchors + i];
        for (size_t c = 1; c < classes.size(); ++c) {
            const float candidate = raw[(4 + c) * anchors + i];
            if (candidate > score) { best = c; score = candidate; }
        }
        if (!std::isfinite(score) || score < threshold) continue;
        const float x = raw[i], y = raw[anchors + i];
        const float w = raw[2 * anchors + i], h = raw[3 * anchors + i];
        if (!std::isfinite(x) || !std::isfinite(y) || !std::isfinite(w) || !std::isfinite(h) || w <= 0 || h <= 0) continue;
        boxes.push_back({x, y, w, h, score, static_cast<int>(best)});
    }
    // This example's network input is 640x640. Mirror Image::preprocess exactly.
    const float gain = std::min(640.0f / image_width, 640.0f / image_height);
    const int resized_width = static_cast<int>(image_width * gain);
    const int resized_height = static_cast<int>(image_height * gain);
    const int pad_x = (640 - resized_width) / 2;
    const int pad_y = (640 - resized_height) / 2;
    for (const auto& box : NMS(boxes, IOU_THRESHOLD)) {
        const int x1 = std::max(0, std::min(image_width, static_cast<int>((box.x - box.w / 2 - pad_x) / gain)));
        const int y1 = std::max(0, std::min(image_height, static_cast<int>((box.y - box.h / 2 - pad_y) / gain)));
        const int x2 = std::max(0, std::min(image_width, static_cast<int>((box.x + box.w / 2 - pad_x) / gain)));
        const int y2 = std::max(0, std::min(image_height, static_cast<int>((box.y + box.h / 2 - pad_y) / gain)));
        if (x2 <= x1 || y2 <= y1) continue;
        detection_struct detection;
        detection.bbox = cv::Rect(x1, y1, x2 - x1, y2 - y1);
        detection.class_id = box.class_id;
        detection.name = classes[box.class_id];
        detection.confidence_score = box.score;
        detections.push_back(detection);
    }
    return 0;
}


float IoU(const BoundingBox& box1, const BoundingBox& box2) {
    float x1_min = box1.x - box1.w / 2.0f;
    float y1_min = box1.y - box1.h / 2.0f;
    float x1_max = box1.x + box1.w / 2.0f;
    float y1_max = box1.y + box1.h / 2.0f;

    float x2_min = box2.x - box2.w / 2.0f;
    float y2_min = box2.y - box2.h / 2.0f;
    float x2_max = box2.x + box2.w / 2.0f;
    float y2_max = box2.y + box2.h / 2.0f;

    float inter_x_min = std::max(x1_min, x2_min);
    float inter_y_min = std::max(y1_min, y2_min);
    float inter_x_max = std::min(x1_max, x2_max);
    float inter_y_max = std::min(y1_max, y2_max);

    float inter_width = inter_x_max - inter_x_min;
    float inter_height = inter_y_max - inter_y_min;

    if (inter_width <= 0 || inter_height <= 0)
        return 0.0f;

    float inter_area = inter_width * inter_height;
    float area1 = (x1_max - x1_min) * (y1_max - y1_min);
    float area2 = (x2_max - x2_min) * (y2_max - y2_min);
    float union_area = area1 + area2 - inter_area;

    return inter_area / union_area;
}

std::vector<BoundingBox> NMS(const std::vector<BoundingBox>& boxes, float iou_threshold) {

    std::vector<BoundingBox> result;

    std::vector<BoundingBox> sorted_boxes = boxes;
    std::sort(sorted_boxes.begin(), sorted_boxes.end(), [](const BoundingBox& a, const BoundingBox& b) {
        return a.score > b.score;
    });

    std::vector<bool> suppressed(sorted_boxes.size(), false);

    for (size_t i = 0; i < sorted_boxes.size(); ++i) {
        if (suppressed[i])
            continue;

        result.push_back(sorted_boxes[i]);

        for (size_t j = i + 1; j < sorted_boxes.size(); ++j) {
            if (suppressed[j])
                continue;
            if (sorted_boxes[i].class_id == sorted_boxes[j].class_id && IoU(sorted_boxes[i], sorted_boxes[j]) > iou_threshold) {
                suppressed[j] = true;
            }
        }
    }
    return result;
}

TritonCommunication::TritonCommunication(std::string triton_address, std::string model_name, std::string model_version, int image_channel, int image_width, int image_height, int class_count) : options(model_name)
{
    triton::client::Error err;
    this->triton_url = triton_address;

    this->options.model_version_ = model_version;

    this->shape = {1, image_channel, image_height, image_width};
    this->input_byte_size = image_channel * image_width * image_height * sizeof(uint16_t) ;
    this->output_byte_size = (class_count + 4) * 8400 * sizeof(uint16_t);

    err = tc::InferenceServerGrpcClient::Create(&(this->client), this->triton_url);
    if (!err.IsOk()) {
        throw std::runtime_error("Create grpc client error: " + err.Message());
    }
    bool live;
    err = client->IsServerReady(&live);
    if (!err.IsOk() || !live) {
        std::cout<< "Triton server is not live !"<<std::endl;
        exit(-1);
    }

    bool model_ready;
    err = client->IsModelReady(&model_ready,model_name,model_version);
    if (!err.IsOk() || !model_ready) {
        std::cerr << "Model:[" << model_name << "] has not been deployed on Triton Server. Triton Server Address:["<<triton_address <<"]"<<std::endl;
        exit(-1);
    }

    std::cout<<"Triton server is LIVE and model is READY!"<<std::endl;
}

void TritonCommunication::infer(uint16_t* image_data)
{
    size_t num_elements = output_byte_size / sizeof(uint16_t);
    tc::Error err;
    tc::InferInput* input0 = nullptr;

    err = tc::InferInput::Create(&input0, "images", shape, "FP16"); // FP16 is the data type of the input tensor.
    if (!err.IsOk()) throw std::runtime_error(err.Message());
    std::shared_ptr<tc::InferInput> input0_ptr;
    input0_ptr.reset(input0);

    err = input0_ptr->AppendRaw((const uint8_t*)image_data, this->input_byte_size);
    if (!err.IsOk()) throw std::runtime_error(err.Message());

    std::vector<tc::InferInput*> inputs = {input0_ptr.get()};

    tc::InferResult* results = nullptr;
    err = client->Infer(&results, options, inputs);
    results_ptr.reset(results);
    if (!err.IsOk()) throw std::runtime_error(err.Message());
    if (!results) throw std::runtime_error("Triton returned no result");

    const uint8_t* output0_data = nullptr;
    size_t output0_byte_size = 0;
    std::vector<uint16_t> result_fp16_raw_data;

    std::string datatype;
    err = results->Datatype("output0", &datatype);
    if (!err.IsOk()) throw std::runtime_error(err.Message());
    if (datatype != "FP16") throw std::runtime_error("Expected FP16 output0");
    err = results->RawData("output0", &output0_data, &output0_byte_size);
    if (!err.IsOk()) throw std::runtime_error(err.Message());
    if (!output0_data || output0_byte_size != output_byte_size)
        throw std::runtime_error("output0 byte size differs from configured 640x640 detection model");
    result_fp16_raw_data.resize(output0_byte_size/sizeof(uint16_t));

    std::memcpy(result_fp16_raw_data.data(), output0_data, output0_byte_size);
    std::vector<float> float32_data(num_elements);

    for (size_t i = 0; i < num_elements; i++) {
        float32_data[i] = float16_to_float32(result_fp16_raw_data[i]);
    }

    output_raw_data = float32_data;

}
