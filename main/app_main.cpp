#include "esp_log.h"
#include "dl_model_base.hpp"
#include "dl_image_define.hpp"
#include "dl_image_preprocessor.hpp"
#include "dl_cls_postprocessor.hpp"
#include "dl_image_jpeg.hpp"
#include "bsp/esp-bsp.h"
#include <esp_system.h>
#include <nvs_flash.h>
#include <string.h>
#include <sys/param.h>

#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "include/BirdPostProcessor.hpp"
#include "esp_camera.h"

// Support IDF 5.x
#ifndef portTICK_RATE_MS
#define portTICK_RATE_MS portTICK_PERIOD_MS
#endif

const char *TAG2 = "bird_cls";

extern const uint8_t birb_jpg_start[] asm("_binary_bluetit_jpg_start");
extern const uint8_t birb_jpg_end[] asm("_binary_bluetit_jpg_end");
extern const uint8_t espdl_model[] asm("_binary_torch_mbnv2_layerwise_equalization_espdl_start");

// Set to true to take a camera picture, else make sure to add an img
#define TAKE_PICTURE true

#if TAKE_PICTURE && ESP_CAMERA_SUPPORTED
#include "camera_pins.h"

// Camera Module pin mapping
static camera_config_t camera_config = {
    .pin_pwdn = PWDN_GPIO_NUM,
    .pin_reset = RESET_GPIO_NUM,
    .pin_xclk = XCLK_GPIO_NUM,
    .pin_sscb_sda = SIOD_GPIO_NUM,
    .pin_sscb_scl = SIOC_GPIO_NUM,

    .pin_d7 = Y9_GPIO_NUM,
    .pin_d6 = Y8_GPIO_NUM,
    .pin_d5 = Y7_GPIO_NUM,
    .pin_d4 = Y6_GPIO_NUM,
    .pin_d3 = Y5_GPIO_NUM,
    .pin_d2 = Y4_GPIO_NUM,
    .pin_d1 = Y3_GPIO_NUM,
    .pin_d0 = Y2_GPIO_NUM,

    .pin_vsync = VSYNC_GPIO_NUM,
    .pin_href = HREF_GPIO_NUM,
    .pin_pclk = PCLK_GPIO_NUM,

    .xclk_freq_hz = 20000000, // XCLK 20MHz or 10MHz for OV2640 double FPS (Experimental)
    .ledc_timer = LEDC_TIMER_0,
    .ledc_channel = LEDC_CHANNEL_0,

    .pixel_format = PIXFORMAT_RGB565, // PIXFORMAT_RGB565 , PIXFORMAT_JPEG
    .frame_size = FRAMESIZE_QVGA,     // [<<320x240>> (QVGA, 4:3);FRAMESIZE_320X320, 240x176 (HQVGA, 15:11); 400x296 (CIF, 50:37)],FRAMESIZE_QVGA,FRAMESIZE_VGA

    .jpeg_quality = 8, // 0-63 lower number means higher quality.  Reduce quality if stack overflow in cam_task
    .fb_count = 2,     // if more than one, i2s runs in continuous mode. Use only with JPEG
    .fb_location = CAMERA_FB_IN_PSRAM,
    .grab_mode = CAMERA_GRAB_WHEN_EMPTY,
    .sccb_i2c_port = 0 // optional
};

static esp_err_t init_camera(void) {
    // Initialize the camera
    esp_err_t err = esp_camera_init(&camera_config);
    if (err != ESP_OK) {
        ESP_LOGE("CAM", "Camera Init Failed");
    }
    return err;
}

static bool capture_image(dl::image::img_t &output_img) {
    ESP_LOGI("CAM", "Taking picture...");
    camera_fb_t *pic = esp_camera_fb_get();
    if (!pic) {
        ESP_LOGE("CAM", "Failed to capture image");
        return false;
    }

    // Use pic->buf to access the image
    ESP_LOGI("CAM", "Picture taken! Its size was: %zu bytes", pic->len);
    ESP_LOGW("image_dim", "Height: %d, Width: %d, Len: %zu", pic->height, pic->width, pic->len);

    // dl::image::jpeg_img_t jpeg_img = {
    //     .data = (uint8_t *)pic->buf,
    //     .width = (int)pic->width,
    //     .height = (int)pic->height,
    //     .data_size = (uint32_t)(pic->len),
    // };

    // Prepare ESP-DL image structs
    dl::image::img_t rgb565_img;
    rgb565_img.data = pic->buf;
    rgb565_img.height = pic->height;
    rgb565_img.width = pic->width;
    rgb565_img.pix_type = dl::image::DL_IMAGE_PIX_TYPE_RGB565;

    output_img.height = pic->height;
    output_img.width = pic->width;
    output_img.pix_type = dl::image::DL_IMAGE_PIX_TYPE_RGB888;
    output_img.data = malloc(pic->height * pic->width * 3); // RGB888: 3 bytes per pixel

    if (!output_img.data) {
        ESP_LOGE("MEM", "Memory allocation failed");
        esp_camera_fb_return(pic);
        return false;
    }

    // Convert using ESP-DL
    int x_min = 0;
    int x_max = 160;
    int y_min = 0;
    int y_max = 160;
    std::vector<int> crop_area = {x_min, y_min, x_max, y_max};
    dl::image::convert_img(rgb565_img, output_img, 0, nullptr, crop_area);

    esp_camera_fb_return(pic);
    return true;
}
#endif // TAKE_PICTURE && ESP_CAMERA_SUPPORTED

static const dl::cls::result_t *run_inference(dl::image::img_t &input_img) {
    char dir[64];
    // TODO: as we are testing multiple models we might want to include them in a smarter way. Is there something like command line arguments?
    snprintf(dir, sizeof(dir), "%s/espdl_models", CONFIG_BSP_SD_MOUNT_POINT);
    dl::Model *model = new dl::Model((const char *)espdl_model, dir);

    uint32_t t0, t1;
    float delta;
    t0 = esp_timer_get_time();

    dl::image::ImagePreprocessor *m_image_preprocessor = new dl::image::ImagePreprocessor(model, {123.675, 116.28, 103.53}, {58.395, 57.12, 57.375});
    m_image_preprocessor->preprocess(input_img);

    model->run();
    const int check = 5;
    BirdPostProcessor m_postprocessor(model, check, std::numeric_limits<float>::lowest(), true);
    std::vector<dl::cls::result_t> &results = m_postprocessor.postprocess();

    t1 = esp_timer_get_time();
    delta = t1 - t0;
    printf("Inference in %8.0f us.", delta);

    const dl::cls::result_t *best_result = nullptr;

    for (auto &res : results) {
        ESP_LOGI(TAG2, "category: %s, score: %f\n", res.cat_name, res.score);
        if (best_result == nullptr || res.score > best_result->score)
        {
            best_result = &res;
        }
    }

    // Free resources
    if (model) {
        delete model;
        model = nullptr;
    }
    if (m_image_preprocessor) {
        delete m_image_preprocessor;
        m_image_preprocessor = nullptr;
    }

    return best_result;
}

extern "C" void app_main(void) {
    dl::image::img_t img;

#if TAKE_PICTURE && ESP_CAMERA_SUPPORTED
    if (init_camera() != ESP_OK || !capture_image(img)) {
        return;
    }
#else
    // If an example image is used
    dl::image::jpeg_img_t jpeg_img = {
        .data = (uint8_t *)birb_jpg_start,
        .width = 160,
        .height = 160,
        .data_size = (uint32_t)(birb_jpg_end - birb_jpg_start),
    };
    img.pix_type = dl::image::DL_IMAGE_PIX_TYPE_RGB888;
    sw_decode_jpeg(jpeg_img, img, true);
#endif

    const auto *best = run_inference(img);
    if (best) {
        ESP_LOGI(TAG2, "Best: %s (score: %f)", best->cat_name, best->score);
    }

    heap_caps_free(img.data);
    ESP_LOGI("MEM", "Free heap at end of loop: %lu bytes", esp_get_free_heap_size());
}
