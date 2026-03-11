#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <omp.h>
#include <pthread.h>

#define ROWS 24
#define COLS 200
#define KERNEL_SIZE 5
#define KERNEL_SUM 273.0
#define MAX_THREADS_TEST 7
#define EPS 1e-9
#define REPEATS 30

static const int kernel[5][5] = {
    {1, 4, 7, 4, 1},
    {4, 16, 26, 16, 4},
    {7, 26, 41, 26, 7},
    {4, 16, 26, 16, 4},
    {1, 4, 7, 4, 1}
};

double grid[ROWS][COLS];
double seq_output[ROWS][COLS];
double omp_static_output[ROWS][COLS];
double omp_dynamic_output[ROWS][COLS];
double pth_output[ROWS][COLS];

typedef struct {
    int start_row;
    int end_row;
    double (*input)[COLS];
    double (*output)[COLS];
} thread_data_t;

double get_time() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

void load_grid(const char *filename) {
    FILE *fp = fopen(filename, "r");
    if (!fp) {
        fprintf(stderr, "Error: could not open %s\n", filename);
        exit(EXIT_FAILURE);
    }

    for (int i = 0; i < ROWS; i++) {
        for (int j = 0; j < COLS; j++) {
            int ok = (j == COLS - 1) ? fscanf(fp, "%lf", &grid[i][j])
                                     : fscanf(fp, "%lf,", &grid[i][j]);
            if (ok != 1) {
                fprintf(stderr, "Error reading grid at row=%d col=%d\n", i, j);
                fclose(fp);
                exit(EXIT_FAILURE);
            }
        }
    }

    fclose(fp);
}

void save_grid(const char *filename, double out[ROWS][COLS]) {
    FILE *fp = fopen(filename, "w");
    if (!fp) {
        fprintf(stderr, "Error: could not write %s\n", filename);
        exit(EXIT_FAILURE);
    }

    for (int i = 0; i < ROWS; i++) {
        for (int j = 0; j < COLS; j++) {
            fprintf(fp, "%.6f", out[i][j]);
            if (j < COLS - 1) fprintf(fp, ",");
        }
        fprintf(fp, "\n");
    }

    fclose(fp);
}

void apply_smoothing_cell(double in[ROWS][COLS], double out[ROWS][COLS], int r, int c) {
    double sum = 0.0;
    int offset = KERNEL_SIZE / 2;

    for (int ki = 0; ki < KERNEL_SIZE; ki++) {
        for (int kj = 0; kj < KERNEL_SIZE; kj++) {
            int ni = r + ki - offset;
            int nj = c + kj - offset;

            if (ni >= 0 && ni < ROWS && nj >= 0 && nj < COLS) {
                sum += in[ni][nj] * kernel[ki][kj];
            }
        }
    }

    out[r][c] = sum / KERNEL_SUM;
}

void smooth_sequential(double in[ROWS][COLS], double out[ROWS][COLS]) {
    for (int r = 0; r < ROWS; r++) {
        for (int c = 0; c < COLS; c++) {
            apply_smoothing_cell(in, out, r, c);
        }
    }
}

void smooth_openmp_static(double in[ROWS][COLS], double out[ROWS][COLS], int threads) {
    omp_set_num_threads(threads);

    #pragma omp parallel for collapse(2) schedule(static)
    for (int r = 0; r < ROWS; r++) {
        for (int c = 0; c < COLS; c++) {
            apply_smoothing_cell(in, out, r, c);
        }
    }
}

void smooth_openmp_dynamic(double in[ROWS][COLS], double out[ROWS][COLS], int threads) {
    omp_set_num_threads(threads);

    #pragma omp parallel for collapse(2) schedule(dynamic, 1)
    for (int r = 0; r < ROWS; r++) {
        for (int c = 0; c < COLS; c++) {
            apply_smoothing_cell(in, out, r, c);
        }
    }
}

void *smooth_thread_func(void *arg) {
    thread_data_t *data = (thread_data_t *)arg;

    for (int r = data->start_row; r < data->end_row; r++) {
        for (int c = 0; c < COLS; c++) {
            apply_smoothing_cell(data->input, data->output, r, c);
        }
    }

    return NULL;
}

void smooth_pthreads(double in[ROWS][COLS], double out[ROWS][COLS], int num_threads) {
    pthread_t *threads = malloc(num_threads * sizeof(pthread_t));
    thread_data_t *thread_data = malloc(num_threads * sizeof(thread_data_t));

    if (!threads || !thread_data) {
        fprintf(stderr, "Error: memory allocation failed for pthread structures\n");
        free(threads);
        free(thread_data);
        exit(EXIT_FAILURE);
    }

    int rows_per_thread = ROWS / num_threads;
    int extra = ROWS % num_threads;
    int start = 0;

    for (int i = 0; i < num_threads; i++) {
        int count = rows_per_thread + (i < extra ? 1 : 0);

        thread_data[i].start_row = start;
        thread_data[i].end_row = start + count;
        thread_data[i].input = in;
        thread_data[i].output = out;

        if (pthread_create(&threads[i], NULL, smooth_thread_func, &thread_data[i]) != 0) {
            fprintf(stderr, "Error: pthread_create failed for thread %d\n", i);
            free(threads);
            free(thread_data);
            exit(EXIT_FAILURE);
        }

        start += count;
    }

    for (int i = 0; i < num_threads; i++) {
        pthread_join(threads[i], NULL);
    }

    free(threads);
    free(thread_data);
}

int compare_outputs(double a[ROWS][COLS], double b[ROWS][COLS], double tol) {
    for (int i = 0; i < ROWS; i++) {
        for (int j = 0; j < COLS; j++) {
            if (fabs(a[i][j] - b[i][j]) > tol) {
                fprintf(stderr, "Mismatch at (%d,%d): %.12f vs %.12f\n",
                        i, j, a[i][j], b[i][j]);
                return 0;
            }
        }
    }
    return 1;
}

double benchmark_seq(double in[ROWS][COLS], double out[ROWS][COLS]) {
    double total = 0.0;
    for (int r = 0; r < REPEATS; r++) {
        double start = get_time();
        smooth_sequential(in, out);
        total += (get_time() - start);
    }
    return total / REPEATS;
}

double benchmark_omp_static(double in[ROWS][COLS], double out[ROWS][COLS], int threads) {
    double total = 0.0;
    for (int r = 0; r < REPEATS; r++) {
        double start = get_time();
        smooth_openmp_static(in, out, threads);
        total += (get_time() - start);
    }
    return total / REPEATS;
}

double benchmark_omp_dynamic(double in[ROWS][COLS], double out[ROWS][COLS], int threads) {
    double total = 0.0;
    for (int r = 0; r < REPEATS; r++) {
        double start = get_time();
        smooth_openmp_dynamic(in, out, threads);
        total += (get_time() - start);
    }
    return total / REPEATS;
}

double benchmark_pthreads(double in[ROWS][COLS], double out[ROWS][COLS], int threads) {
    double total = 0.0;
    for (int r = 0; r < REPEATS; r++) {
        double start = get_time();
        smooth_pthreads(in, out, threads);
        total += (get_time() - start);
    }
    return total / REPEATS;
}

void write_performance_csv(const char *filename,
                           int t_counts[],
                           int ntests,
                           double seq_time,
                           double omp_static_times[],
                           double omp_dynamic_times[],
                           double pth_times[]) {
    FILE *fp = fopen(filename, "w");
    if (!fp) {
        fprintf(stderr, "Error: could not write %s\n", filename);
        exit(EXIT_FAILURE);
    }

    fprintf(fp,
            "threads,sequential_time,openmp_static_time,openmp_dynamic_time,pthreads_time,"
            "speedup_openmp_static,speedup_openmp_dynamic,speedup_pthreads,"
            "efficiency_openmp_static,efficiency_openmp_dynamic,efficiency_pthreads\n");

    for (int i = 0; i < ntests; i++) {
        double s1 = seq_time / omp_static_times[i];
        double s2 = seq_time / omp_dynamic_times[i];
        double s3 = (pth_times[i] > 0.0) ? (seq_time / pth_times[i]) : -1.0;

        double e1 = s1 / t_counts[i];
        double e2 = s2 / t_counts[i];
        double e3 = (pth_times[i] > 0.0) ? (s3 / t_counts[i]) : -1.0;

        fprintf(fp, "%d,%.9f,%.9f,%.9f,%.9f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f\n",
                t_counts[i],
                seq_time,
                omp_static_times[i],
                omp_dynamic_times[i],
                pth_times[i],
                s1, s2, s3,
                e1, e2, e3);
    }

    fclose(fp);
}

int main() {
    load_grid("grid_z.csv");

    int t_counts[MAX_THREADS_TEST] = {1, 2, 4, 8, 12, 16, 24};
    int num_tests = MAX_THREADS_TEST;

    double omp_static_times[MAX_THREADS_TEST];
    double omp_dynamic_times[MAX_THREADS_TEST];
    double pth_times[MAX_THREADS_TEST];

    double seq_time = benchmark_seq(grid, seq_output);

    for (int i = 0; i < num_tests; i++) {
        omp_static_times[i] = benchmark_omp_static(grid, omp_static_output, t_counts[i]);
    }

    for (int i = 0; i < num_tests; i++) {
        omp_dynamic_times[i] = benchmark_omp_dynamic(grid, omp_dynamic_output, t_counts[i]);
    }

    for (int i = 0; i < num_tests; i++) {
        if (t_counts[i] > ROWS) {
            pth_times[i] = -1.0;
            continue;
        }
        pth_times[i] = benchmark_pthreads(grid, pth_output, t_counts[i]);
    }

    int ok_static = compare_outputs(seq_output, omp_static_output, EPS);
    int ok_dynamic = compare_outputs(seq_output, omp_dynamic_output, EPS);
    int ok_pth = compare_outputs(seq_output, pth_output, EPS);

    printf("\n=== Correctness Check ===\n");
    printf("OpenMP static  : %s\n", ok_static ? "PASS" : "FAIL");
    printf("OpenMP dynamic : %s\n", ok_dynamic ? "PASS" : "FAIL");
    printf("pThreads       : %s\n", ok_pth ? "PASS" : "FAIL");

    printf("\n=== Performance Comparison (average of %d runs, grid 24x200) ===\n", REPEATS);
    printf("Sequential baseline: %.9f s\n\n", seq_time);

    printf("%-8s | %-15s | %-15s | %-15s | %-12s | %-12s | %-12s\n",
           "Threads", "OMP Static (s)", "OMP Dynamic (s)", "pThreads (s)",
           "Spd OMP-S", "Spd OMP-D", "Spd PTH");
    printf("-----------------------------------------------------------------------------------------------------\n");

    for (int i = 0; i < num_tests; i++) {
        double sp_static = seq_time / omp_static_times[i];
        double sp_dynamic = seq_time / omp_dynamic_times[i];

        printf("%-8d | %-15.9f | %-15.9f | ",
               t_counts[i], omp_static_times[i], omp_dynamic_times[i]);

        if (pth_times[i] > 0.0) {
            double sp_pth = seq_time / pth_times[i];
            printf("%-15.9f | %-12.4f | %-12.4f | %-12.4f\n",
                   pth_times[i], sp_static, sp_dynamic, sp_pth);
        } else {
            printf("%-15s | %-12.4f | %-12.4f | %-12s\n",
                   "N/A", sp_static, sp_dynamic, "N/A");
        }
    }

    save_grid("smoothed_seq.csv", seq_output);
    write_performance_csv("performance_results.csv",
                          t_counts, num_tests, seq_time,
                          omp_static_times, omp_dynamic_times, pth_times);

    printf("\nFiles generated:\n");
    printf(" - smoothed_seq.csv\n");
    printf(" - performance_results.csv\n");

    return 0;
}
