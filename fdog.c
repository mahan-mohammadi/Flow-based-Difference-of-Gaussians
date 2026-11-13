#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <omp.h>

/*
 * Flow-based Difference of Gaussians (FDoG)
 *
 * Usage:
 *   ./fdog input.ppm output.ppm [threads]
 *
 * Input/output: binary PPM (P6), max value 255.
 *
 * Compile:
 *   gcc -O3 -march=native -fopenmp fdog.c -lm -o fdog
 *
 * Pipeline:
 *   1. Convert RGB to grayscale.
 *   2. Compute image gradients with Sobel.
 *   3. Build and smooth the structure tensor.
 *   4. Extract the Edge Tangent Flow (ETF).
 *   5. Apply a DoG filter perpendicular to the ETF.
 *   6. Smooth the response along the ETF.
 *   7. Normalize the response (by its RMS), apply a soft threshold,
 *      and write a grayscale image.
 */

#define DEFAULT_THREADS 4

/* FDoG parameters. */
#define SIGMA_C       2.0f   /* Structure-tensor smoothing. */
#define SIGMA_E       1.0f   /* DoG width across the edge. */
#define SIGMA_S       1.6f   /* Wider Gaussian in the DoG. */
#define RHO           0.99f  /* Relative weight of the wider Gaussian. */
#define SIGMA_M       3.0f   /* Smoothing distance along the edge flow. */

/*
 * Threshold parameters. Both are in units of the response RMS, so the
 * result does not depend on the contrast of the input image.
 *
 *   EPSILON: lines appear where the normalized response drops below it.
 *            Closer to 0 gives more lines, closer to -1 gives fewer.
 *   PHI:     steepness of the black/white transition.
 */
#define PHI           4.0f
#define EPSILON      -0.02f

typedef struct {
    int width;
    int height;
    unsigned char *pixels;
} Image;

typedef struct {
    float x;
    float y;
} Vector2;

/* ---------- PPM loading and saving ---------- */

static int read_ppm_token(FILE *file, char *token, size_t capacity)
{
    int c;
    size_t length = 0;

    for (;;) {
        c = fgetc(file);
        if (c == EOF)
            return 0;

        if (c == '#') {
            while ((c = fgetc(file)) != EOF && c != '\n')
                ;
            continue;
        }

        if (c > ' ')
            break;
    }

    do {
        if (length + 1 < capacity)
            token[length++] = (char)c;
        c = fgetc(file);
    } while (c != EOF && c > ' ');

    token[length] = '\0';
    return 1;
}

static Image *load_ppm(const char *filename)
{
    FILE *file = fopen(filename, "rb");
    if (!file) {
        perror("Could not open input image");
        return NULL;
    }

    char token[64];

    if (!read_ppm_token(file, token, sizeof(token)) ||
        strcmp(token, "P6") != 0) {
        fprintf(stderr, "Error: input must be a binary PPM (P6) image.\n");
        fclose(file);
        return NULL;
    }

    if (!read_ppm_token(file, token, sizeof(token))) {
        fprintf(stderr, "Error: missing image width.\n");
        fclose(file);
        return NULL;
    }
    int width = atoi(token);

    if (!read_ppm_token(file, token, sizeof(token))) {
        fprintf(stderr, "Error: missing image height.\n");
        fclose(file);
        return NULL;
    }
    int height = atoi(token);

    if (!read_ppm_token(file, token, sizeof(token)) || atoi(token) != 255) {
        fprintf(stderr, "Error: only PPM images with max value 255 are supported.\n");
        fclose(file);
        return NULL;
    }

    if (width <= 0 || height <= 0 ||
        (size_t)width > (size_t)-1 / (size_t)height / 3) {
        fprintf(stderr, "Error: invalid image dimensions.\n");
        fclose(file);
        return NULL;
    }

    Image *image = malloc(sizeof(*image));
    if (!image) {
        perror("malloc");
        fclose(file);
        return NULL;
    }

    size_t bytes = (size_t)width * (size_t)height * 3;
    image->pixels = malloc(bytes);
    if (!image->pixels) {
        perror("malloc");
        free(image);
        fclose(file);
        return NULL;
    }

    image->width = width;
    image->height = height;

    if (fread(image->pixels, 1, bytes, file) != bytes) {
        fprintf(stderr, "Error: could not read all image pixels.\n");
        free(image->pixels);
        free(image);
        fclose(file);
        return NULL;
    }

    fclose(file);
    return image;
}

static int save_ppm(const char *filename, const Image *image)
{
    FILE *file = fopen(filename, "wb");
    if (!file) {
        perror("Could not open output image");
        return 0;
    }

    fprintf(file, "P6\n%d %d\n255\n", image->width, image->height);

    size_t bytes = (size_t)image->width * (size_t)image->height * 3;
    int success = fwrite(image->pixels, 1, bytes, file) == bytes;

    if (fclose(file) != 0)
        success = 0;

    return success;
}

/* ---------- Small utility functions ---------- */

static int clamp_int(int value, int low, int high)
{
    if (value < low) return low;
    if (value > high) return high;
    return value;
}

static float clamp_float(float value, float low, float high)
{
    if (value < low) return low;
    if (value > high) return high;
    return value;
}

static float sample_bilinear(
    const float *image, int width, int height, float x, float y)
{
    x = clamp_float(x, 0.0f, (float)(width - 1));
    y = clamp_float(y, 0.0f, (float)(height - 1));

    int x0 = (int)floorf(x);
    int y0 = (int)floorf(y);
    int x1 = clamp_int(x0 + 1, 0, width - 1);
    int y1 = clamp_int(y0 + 1, 0, height - 1);

    float tx = x - (float)x0;
    float ty = y - (float)y0;

    float a = image[(size_t)y0 * width + x0];
    float b = image[(size_t)y0 * width + x1];
    float c = image[(size_t)y1 * width + x0];
    float d = image[(size_t)y1 * width + x1];

    float top = a + tx * (b - a);
    float bottom = c + tx * (d - c);
    return top + ty * (bottom - top);
}

/*
 * An eigenvector has no inherent sign, so neighboring flow vectors can
 * point in opposite directions. Flip a vector so that it agrees with a
 * reference vector before the two are blended together.
 */
static Vector2 align_to(Vector2 v, Vector2 reference)
{
    if (v.x * reference.x + v.y * reference.y < 0.0f) {
        v.x = -v.x;
        v.y = -v.y;
    }
    return v;
}

static Vector2 sample_flow(
    const Vector2 *flow, int width, int height, float x, float y)
{
    x = clamp_float(x, 0.0f, (float)(width - 1));
    y = clamp_float(y, 0.0f, (float)(height - 1));

    int x0 = (int)floorf(x);
    int y0 = (int)floorf(y);
    int x1 = clamp_int(x0 + 1, 0, width - 1);
    int y1 = clamp_int(y0 + 1, 0, height - 1);

    float tx = x - (float)x0;
    float ty = y - (float)y0;

    /* Align the neighbors to the first sample before interpolating,
       otherwise opposite signs cancel and the direction becomes bogus. */
    Vector2 a = flow[(size_t)y0 * width + x0];
    Vector2 b = align_to(flow[(size_t)y0 * width + x1], a);
    Vector2 c = align_to(flow[(size_t)y1 * width + x0], a);
    Vector2 d = align_to(flow[(size_t)y1 * width + x1], a);

    Vector2 result;
    result.x = (1.0f - ty) * ((1.0f - tx) * a.x + tx * b.x)
             + ty * ((1.0f - tx) * c.x + tx * d.x);
    result.y = (1.0f - ty) * ((1.0f - tx) * a.y + tx * b.y)
             + ty * ((1.0f - tx) * c.y + tx * d.y);

    float length = sqrtf(result.x * result.x + result.y * result.y);
    if (length > 1.0e-6f) {
        result.x /= length;
        result.y /= length;
    } else {
        result.x = 1.0f;
        result.y = 0.0f;
    }

    return result;
}

/* ---------- Grayscale and Gaussian blur ---------- */

static float *rgb_to_grayscale(const Image *image)
{
    size_t count = (size_t)image->width * (size_t)image->height;
    float *gray = malloc(count * sizeof(*gray));
    if (!gray) {
        perror("malloc");
        return NULL;
    }

    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < count; i++) {
        float r = image->pixels[3 * i + 0];
        float g = image->pixels[3 * i + 1];
        float b = image->pixels[3 * i + 2];
        gray[i] = (0.299f * r + 0.587f * g + 0.114f * b) / 255.0f;
    }

    return gray;
}

static float *create_gaussian_kernel(float sigma, int *radius_out)
{
    int radius = (int)ceilf(3.0f * sigma);
    int size = 2 * radius + 1;

    float *kernel = malloc((size_t)size * sizeof(*kernel));
    if (!kernel) {
        perror("malloc");
        return NULL;
    }

    float sum = 0.0f;
    for (int i = -radius; i <= radius; i++) {
        float value = expf(-(float)(i * i) / (2.0f * sigma * sigma));
        kernel[i + radius] = value;
        sum += value;
    }

    for (int i = 0; i < size; i++)
        kernel[i] /= sum;

    *radius_out = radius;
    return kernel;
}

static void gaussian_blur(
    const float *input,
    float *output,
    float *temporary,
    int width,
    int height,
    const float *kernel,
    int radius)
{
    #pragma omp parallel for schedule(static)
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            float sum = 0.0f;

            for (int offset = -radius; offset <= radius; offset++) {
                int sx = clamp_int(x + offset, 0, width - 1);
                sum += input[(size_t)y * width + sx]
                     * kernel[offset + radius];
            }

            temporary[(size_t)y * width + x] = sum;
        }
    }

    #pragma omp parallel for schedule(static)
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            float sum = 0.0f;

            for (int offset = -radius; offset <= radius; offset++) {
                int sy = clamp_int(y + offset, 0, height - 1);
                sum += temporary[(size_t)sy * width + x]
                     * kernel[offset + radius];
            }

            output[(size_t)y * width + x] = sum;
        }
    }
}

/* ---------- Gradients and structure tensor ---------- */

static void calculate_gradients(
    const float *gray, float *gx, float *gy, int width, int height)
{
    #pragma omp parallel for schedule(static)
    for (int y = 0; y < height; y++) {
        int ym = clamp_int(y - 1, 0, height - 1);
        int yp = clamp_int(y + 1, 0, height - 1);

        for (int x = 0; x < width; x++) {
            int xm = clamp_int(x - 1, 0, width - 1);
            int xp = clamp_int(x + 1, 0, width - 1);

            float tl = gray[(size_t)ym * width + xm];
            float tc = gray[(size_t)ym * width + x];
            float tr = gray[(size_t)ym * width + xp];
            float ml = gray[(size_t)y * width + xm];
            float mr = gray[(size_t)y * width + xp];
            float bl = gray[(size_t)yp * width + xm];
            float bc = gray[(size_t)yp * width + x];
            float br = gray[(size_t)yp * width + xp];

            size_t i = (size_t)y * width + x;

            /* Sobel derivatives, normalized by 8. */
            gx[i] = (-tl + tr - 2.0f * ml + 2.0f * mr - bl + br) / 8.0f;
            gy[i] = (-tl - 2.0f * tc - tr + bl + 2.0f * bc + br) / 8.0f;
        }
    }
}

static void build_structure_tensor(
    const float *gx,
    const float *gy,
    float *tensor_xx,
    float *tensor_xy,
    float *tensor_yy,
    size_t count)
{
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < count; i++) {
        tensor_xx[i] = gx[i] * gx[i];
        tensor_xy[i] = gx[i] * gy[i];
        tensor_yy[i] = gy[i] * gy[i];
    }
}

/*
 * The smoothed structure tensor is:
 *
 *       [ A  B ]       A = G * Ix^2
 *   J = [      ]       B = G * IxIy
 *       [ B  C ]       C = G * Iy^2
 *
 * Its dominant eigenvector estimates the gradient direction.
 * The perpendicular vector is the edge tangent direction.
 */
static void calculate_edge_tangent_flow(
    const float *tensor_xx,
    const float *tensor_xy,
    const float *tensor_yy,
    Vector2 *flow,
    size_t count)
{
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < count; i++) {
        float a = tensor_xx[i];
        float b = tensor_xy[i];
        float c = tensor_yy[i];

        float angle = 0.5f * atan2f(2.0f * b, a - c);

        /* Gradient direction is (cos(angle), sin(angle)).
           Rotate it by 90 degrees to obtain the tangent. */
        flow[i].x = -sinf(angle);
        flow[i].y =  cosf(angle);
    }
}

/* ---------- First FDoG pass: filter across the flow ---------- */

static float *create_dog_kernel(
    float sigma_e, float sigma_s, float rho, int *radius_out)
{
    int radius = (int)ceilf(3.0f * sigma_s);
    int size = 2 * radius + 1;

    float *kernel = malloc((size_t)size * sizeof(*kernel));
    if (!kernel) {
        perror("malloc");
        return NULL;
    }

    float sum_e = 0.0f;
    float sum_s = 0.0f;

    for (int i = -radius; i <= radius; i++) {
        float x = (float)i;
        sum_e += expf(-(x * x) / (2.0f * sigma_e * sigma_e));
        sum_s += expf(-(x * x) / (2.0f * sigma_s * sigma_s));
    }

    for (int i = -radius; i <= radius; i++) {
        float x = (float)i;
        float ge = expf(-(x * x) / (2.0f * sigma_e * sigma_e)) / sum_e;
        float gs = expf(-(x * x) / (2.0f * sigma_s * sigma_s)) / sum_s;
        kernel[i + radius] = ge - rho * gs;
    }

    *radius_out = radius;
    return kernel;
}

/*
 * At each pixel, the edge tangent is t = (tx, ty).
 * The gradient/normal direction is n = (-ty, tx).
 * We sample the image along n and apply a 1-D DoG kernel.
 */
static void filter_across_edges(
    const float *gray,
    const Vector2 *flow,
    float *response,
    int width,
    int height,
    const float *dog_kernel,
    int radius)
{
    #pragma omp parallel for schedule(static)
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            size_t i = (size_t)y * width + x;
            Vector2 tangent = flow[i];

            float nx = -tangent.y;
            float ny =  tangent.x;
            float sum = 0.0f;

            for (int offset = -radius; offset <= radius; offset++) {
                float sx = (float)x + (float)offset * nx;
                float sy = (float)y + (float)offset * ny;

                sum += sample_bilinear(gray, width, height, sx, sy)
                     * dog_kernel[offset + radius];
            }

            response[i] = sum;
        }
    }
}

/* ---------- Second FDoG pass: smooth along the flow ---------- */

/*
 * Trace the ETF in one direction from (x,y), sampling the first-pass
 * response at each step. The tangent sign is corrected at every step
 * because an eigenvector has no inherent positive/negative direction.
 */
static float trace_one_direction(
    const float *response,
    const Vector2 *flow,
    int width,
    int height,
    float x,
    float y,
    int direction,
    int steps,
    const float *weights,
    float *weight_sum_out)
{
    Vector2 previous = sample_flow(flow, width, height, x, y);
    float sum = 0.0f;
    float weight_sum = 0.0f;

    for (int step = 1; step <= steps; step++) {
        Vector2 tangent = sample_flow(flow, width, height, x, y);

        float dot = tangent.x * previous.x + tangent.y * previous.y;
        if (dot < 0.0f) {
            tangent.x = -tangent.x;
            tangent.y = -tangent.y;
        }

        x += (float)direction * tangent.x;
        y += (float)direction * tangent.y;

        if (x < 0.0f || x > (float)(width - 1) ||
            y < 0.0f || y > (float)(height - 1))
            break;

        sum += sample_bilinear(response, width, height, x, y)
             * weights[step];
        weight_sum += weights[step];

        previous = tangent;
    }

    *weight_sum_out = weight_sum;
    return sum;
}

static float *create_flow_weights(float sigma_m, int *steps_out)
{
    int steps = (int)ceilf(3.0f * sigma_m);
    float *weights = malloc((size_t)(steps + 1) * sizeof(*weights));

    if (!weights) {
        perror("malloc");
        return NULL;
    }

    weights[0] = 1.0f;

    for (int i = 1; i <= steps; i++) {
        float x = (float)i;
        weights[i] = expf(-(x * x) / (2.0f * sigma_m * sigma_m));
    }

    *steps_out = steps;
    return weights;
}

/*
 * Line-integral convolution along the ETF.
 * Each output pixel independently traces its local flow in both
 * directions, so this entire pass is parallelized with OpenMP.
 */
static void filter_along_edges(
    const float *input,
    const Vector2 *flow,
    float *output,
    int width,
    int height,
    const float *weights,
    int steps)
{
    #pragma omp parallel for schedule(static)
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            size_t i = (size_t)y * width + x;

            float weighted_sum = input[i];
            float weight_sum = weights[0];

            float forward_weight = 0.0f;
            float backward_weight = 0.0f;

            float forward = trace_one_direction(
                input, flow, width, height,
                (float)x, (float)y, 1, steps, weights, &forward_weight);

            float backward = trace_one_direction(
                input, flow, width, height,
                (float)x, (float)y, -1, steps, weights, &backward_weight);

            weighted_sum += forward + backward;
            weight_sum += forward_weight + backward_weight;

            output[i] = weighted_sum / weight_sum;
        }
    }
}

/* ---------- Final normalization and soft threshold ---------- */

/*
 * The raw DoG response is small (a strong line gives roughly -0.07 on a
 * [0,1] grayscale image), so a fixed steepness applied to the raw value
 * produces gray lines on a white page. Dividing by the RMS of the
 * response first makes EPSILON and PHI independent of image contrast.
 *
 *   result = 1                                if h >= EPSILON
 *            1 + tanh(PHI * (h - EPSILON))    otherwise
 *
 * where h = response / rms(response).
 */
static void apply_soft_threshold(
    const float *response,
    unsigned char *output,
    size_t count)
{
    double sum_squares = 0.0;

    #pragma omp parallel for reduction(+:sum_squares) schedule(static)
    for (size_t i = 0; i < count; i++)
        sum_squares += (double)response[i] * (double)response[i];

    float scale = (float)sqrt(sum_squares / (double)count);
    if (scale < 1.0e-6f)
        scale = 1.0e-6f;

    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < count; i++) {
        float h = response[i] / scale;
        float result;

        if (h >= EPSILON)
            result = 1.0f;
        else
            result = 1.0f + tanhf(PHI * (h - EPSILON));

        int pixel = (int)lrintf(clamp_float(result, 0.0f, 1.0f) * 255.0f);

        output[3 * i + 0] = (unsigned char)pixel;
        output[3 * i + 1] = (unsigned char)pixel;
        output[3 * i + 2] = (unsigned char)pixel;
    }
}

/* ---------- Main ---------- */

int main(int argc, char **argv)
{
    if (argc < 3 || argc > 4) {
        fprintf(stderr, "Usage: %s input.ppm output.ppm [threads]\n", argv[0]);
        return EXIT_FAILURE;
    }

    int threads = DEFAULT_THREADS;

    if (argc == 4) {
        char *end = NULL;
        long parsed = strtol(argv[3], &end, 10);

        if (!end || *end != '\0' || parsed < 1 || parsed > 1024) {
            fprintf(stderr, "Error: thread count must be between 1 and 1024.\n");
            return EXIT_FAILURE;
        }

        threads = (int)parsed;
    }

    omp_set_dynamic(0);
    omp_set_num_threads(threads);

    Image *input = load_ppm(argv[1]);
    if (!input)
        return EXIT_FAILURE;

    size_t count = (size_t)input->width * (size_t)input->height;

    Image output = {
        .width = input->width,
        .height = input->height,
        .pixels = malloc(count * 3)
    };

    float *gray = NULL;
    float *gx = NULL, *gy = NULL;
    float *tensor_xx = NULL, *tensor_xy = NULL, *tensor_yy = NULL;
    float *response_across = NULL, *response_along = NULL;
    float *tensor_temp = NULL;
    float *smooth_xx = NULL, *smooth_xy = NULL, *smooth_yy = NULL;
    float *kernel_tensor = NULL, *kernel_dog = NULL, *flow_weights = NULL;
    Vector2 *flow = NULL;

    if (!output.pixels) {
        perror("malloc");
        goto cleanup_error;
    }

    gray = rgb_to_grayscale(input);
    gx = malloc(count * sizeof(*gx));
    gy = malloc(count * sizeof(*gy));

    tensor_xx = malloc(count * sizeof(*tensor_xx));
    tensor_xy = malloc(count * sizeof(*tensor_xy));
    tensor_yy = malloc(count * sizeof(*tensor_yy));
    tensor_temp = malloc(count * sizeof(*tensor_temp));
    smooth_xx = malloc(count * sizeof(*smooth_xx));
    smooth_xy = malloc(count * sizeof(*smooth_xy));
    smooth_yy = malloc(count * sizeof(*smooth_yy));

    flow = malloc(count * sizeof(*flow));
    response_across = malloc(count * sizeof(*response_across));
    response_along = malloc(count * sizeof(*response_along));

    if (!gray || !gx || !gy || !tensor_xx || !tensor_xy ||
        !tensor_yy || !tensor_temp || !smooth_xx || !smooth_xy ||
        !smooth_yy || !flow ||
        !response_across || !response_along) {
        perror("malloc");
        goto cleanup_error;
    }

    int tensor_radius;
    kernel_tensor = create_gaussian_kernel(SIGMA_C, &tensor_radius);
    if (!kernel_tensor)
        goto cleanup_error;

    int dog_radius;
    kernel_dog = create_dog_kernel(SIGMA_E, SIGMA_S, RHO, &dog_radius);
    if (!kernel_dog)
        goto cleanup_error;

    int flow_steps;
    flow_weights = create_flow_weights(SIGMA_M, &flow_steps);
    if (!flow_weights)
        goto cleanup_error;

    printf("Image: %d x %d\n", input->width, input->height);
    printf("Threads: %d\n", threads);
    printf("FDoG parameters: sigma_c=%.2f, sigma_e=%.2f, sigma_s=%.2f, "
           "rho=%.2f, sigma_m=%.2f, phi=%.2f, epsilon=%.2f\n",
           SIGMA_C, SIGMA_E, SIGMA_S, RHO, SIGMA_M, PHI, EPSILON);

    double start = omp_get_wtime();

    calculate_gradients(gray, gx, gy, input->width, input->height);

    build_structure_tensor(
        gx, gy, tensor_xx, tensor_xy, tensor_yy, count);

    /* Smooth all three structure-tensor components independently. */
    gaussian_blur(tensor_xx, smooth_xx, tensor_temp,
                  input->width, input->height, kernel_tensor, tensor_radius);
    gaussian_blur(tensor_xy, smooth_xy, tensor_temp,
                  input->width, input->height, kernel_tensor, tensor_radius);
    gaussian_blur(tensor_yy, smooth_yy, tensor_temp,
                  input->width, input->height, kernel_tensor, tensor_radius);

    calculate_edge_tangent_flow(
        smooth_xx, smooth_xy, smooth_yy, flow, count);

    filter_across_edges(
        gray, flow, response_across,
        input->width, input->height, kernel_dog, dog_radius);

    filter_along_edges(
        response_across, flow, response_along,
        input->width, input->height, flow_weights, flow_steps);

    apply_soft_threshold(response_along, output.pixels, count);

    double end = omp_get_wtime();

    if (!save_ppm(argv[2], &output)) {
        fprintf(stderr, "Error: could not save output image.\n");
        goto cleanup_error;
    }

    printf("Processing time: %.4f seconds\n", end - start);
    printf("Saved: %s\n", argv[2]);

    free(input->pixels);
    free(input);
    free(output.pixels);
    free(gray);
    free(gx);
    free(gy);
    free(tensor_xx);
    free(tensor_xy);
    free(tensor_yy);
    free(tensor_temp);
    free(smooth_xx);
    free(smooth_xy);
    free(smooth_yy);
    free(flow);
    free(response_across);
    free(response_along);
    free(kernel_tensor);
    free(kernel_dog);
    free(flow_weights);

    return EXIT_SUCCESS;

cleanup_error:
    if (input) {
        free(input->pixels);
        free(input);
    }
    free(output.pixels);
    free(gray);
    free(gx);
    free(gy);
    free(tensor_xx);
    free(tensor_xy);
    free(tensor_yy);
    free(tensor_temp);
    free(smooth_xx);
    free(smooth_xy);
    free(smooth_yy);
    free(flow);
    free(response_across);
    free(response_along);
    free(kernel_tensor);
    free(kernel_dog);
    free(flow_weights);
    return EXIT_FAILURE;
}
