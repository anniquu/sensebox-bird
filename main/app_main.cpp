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

// support IDF 5.x
#ifndef portTICK_RATE_MS
#define portTICK_RATE_MS portTICK_PERIOD_MS
#endif
// Set to true to take a camera picture.
#define TAKE_PICTURE false
#include "esp_camera.h"

extern const uint8_t cat_jpg_start[] asm("_binary_bluetit_jpg_start");
extern const uint8_t cat_jpg_end[] asm("_binary_bluetit_jpg_end");
extern const uint8_t espdl_model[] asm("_binary_torch_mbnv2_layerwise_equalization_espdl_start");

const char *TAG2 = "bird_cls";
#if TAKE_PICTURE
#define CAMERA_MODEL_XIAO_ESP32S3 // Has PSRAM

#define PWDN_GPIO_NUM -1
#define RESET_GPIO_NUM -1
#define XCLK_GPIO_NUM 10
#define SIOD_GPIO_NUM 40
#define SIOC_GPIO_NUM 39

#define Y9_GPIO_NUM 48
#define Y8_GPIO_NUM 11
#define Y7_GPIO_NUM 12
#define Y6_GPIO_NUM 14
#define Y5_GPIO_NUM 16
#define Y4_GPIO_NUM 18
#define Y3_GPIO_NUM 17
#define Y2_GPIO_NUM 15
#define VSYNC_GPIO_NUM 38
#define HREF_GPIO_NUM 47
#define PCLK_GPIO_NUM 13
static const char *TAG = "example:take_picture";

#if ESP_CAMERA_SUPPORTED
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

    // XCLK 20MHz or 10MHz for OV2640 double FPS (Experimental)
    .xclk_freq_hz = 20000000,
    .ledc_timer = LEDC_TIMER_0,
    .ledc_channel = LEDC_CHANNEL_0,

    // Change to PIXFORMAT_RGB888
    .pixel_format = PIXFORMAT_JPEG,  // YUV422,GRAYSCALE,RGB565,JPEG
    .frame_size = FRAMESIZE_240X240, // QQVGA-UXGA Do not use sizes above QVGA when not JPEG

    .jpeg_quality = 12, // 0-63 lower number means higher quality
    .fb_count = 1,      // if more than one, i2s runs in continuous mode. Use only with JPEG
    .fb_location = CAMERA_FB_IN_PSRAM,
    .grab_mode = CAMERA_GRAB_WHEN_EMPTY,
};

static esp_err_t init_camera(void)
{
    // initialize the camera
    esp_err_t err = esp_camera_init(&camera_config);
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "Camera Init Failed");
        return err;
    }

    return ESP_OK;
}
#endif
#endif
extern "C" void app_main(void)
{
#if TAKE_PICTURE
#if ESP_CAMERA_SUPPORTED
    if (ESP_OK != init_camera()) {
        return;
    }

    ESP_LOGI(TAG, "Taking picture...");
    camera_fb_t *pic = esp_camera_fb_get();

    // use pic->buf to access the image
    ESP_LOGI(TAG, "Picture taken! Its size was: %zu bytes", pic->len);
    esp_camera_fb_return(pic);

#else
    ESP_LOGE(TAG, "Camera support is not available for this chip");
    return;
#endif
#endif

    // dl::image::jpeg_img_t jpeg_img = {
    //     .data = (uint8_t *) pic->buf,
    //     .width = (int) pic->width,
    //     .height = (int) pic->height,
    //     .data_size = (uint32_t)(pic->len),
    // };
    dl::image::jpeg_img_t jpeg_img = {
        .data = (uint8_t *)cat_jpg_start,
        .width = 160,
        .height = 160,
        .data_size = (uint32_t)(cat_jpg_end - cat_jpg_start),
    };

    dl::image::img_t img;
    img.pix_type = dl::image::DL_IMAGE_PIX_TYPE_RGB888;
    sw_decode_jpeg(jpeg_img, img, true);

    char dir[64];
    // TODO: as we are testing multiple models we might want to include them in a smarter way. Is there something like command line arguments?
    snprintf(dir, sizeof(dir), "%s/espdl_models", CONFIG_BSP_SD_MOUNT_POINT);
    dl::Model* model = new dl::Model((const char *)espdl_model, dir);

    uint32_t t0, t1;
    float delta;
    t0 = esp_timer_get_time();

    dl::image::ImagePreprocessor* m_image_preprocessor = new dl::image::ImagePreprocessor(model, {123.675, 116.28, 103.53}, {58.395, 57.12, 57.375});
    m_image_preprocessor->preprocess(img);

    model->run();
    const int check = 5;
    BirdPostProcessor m_postprocessor(model, check, std::numeric_limits<float>::lowest(), true);
    std::vector<dl::cls::result_t> &results = m_postprocessor.postprocess();

    t1 = esp_timer_get_time();
    delta = t1 - t0;
    printf("Inference in %8.0f us.", delta);
    
    for (const auto &res : results) {
         ESP_LOGI(TAG2, "category: %s, score: %f\n", res.cat_name, res.score);
    }
    delete model;
    heap_caps_free(img.data);
}
