#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>
#include <pthread.h>

#define ROWS 24
#define COLS 200
#define KERNEL_SIZE 5
#define KERNEL_SUM 273
#define ITERATIONS 5000

int kernel[5][5] = {
    {1, 4, 7, 4, 1},
    {4, 16, 26, 16, 4},
    {7, 26, 41, 26, 7},
    {4, 16, 26, 16, 4},
    {1, 4, 7, 4, 1}
};

double grid[ROWS][COLS];
double output[ROWS][COLS];

void load_grid(const char* filename) {
    FILE* fp = fopen(filename, "r");
    if (!fp) { 
        fprintf(stderr, "Error: Could not open %s\n", filename); 
        exit(1); 
    }
    
    for (int i = 0; i < ROWS; i++) {
        for (int j = 0; j < COLS; j++) {
            int result;
            
            if (j == COLS - 1) {
                result = fscanf(fp, "%lf\n", &grid[i][j]);
            } else {
                result = fscanf(fp, "%lf,", &grid[i][j]);
            }

            if (result != 1) {
                fprintf(stderr, "Error reading data at row %d, col %d\n", i, j);
                fclose(fp);
                exit(1);
            }
        }
    }
    fclose(fp);
}

void apply_smoothing(int r, int c) {
    double sum = 0;
    int offset = KERNEL_SIZE / 2;
    for (int ki = 0; ki < KERNEL_SIZE; ki++) {
        for (int kj = 0; kj < KERNEL_SIZE; kj++) {
            int ni = r + ki - offset;
            int nj = c + kj - offset;
            if (ni >= 0 && ni < ROWS && nj >= 0 && nj < COLS)
                sum += grid[ni][nj] * kernel[ki][kj];
        }
    }
    output[r][c] = sum / KERNEL_SUM;
}

void smooth_in_sequential() {
    for (int r = 0; r < ROWS; r++)
        for (int c = 0; c < COLS; c++)
            apply_smoothing(r, c);
}

void smooth_openmp(int threads) {
    omp_set_num_threads(threads);
    #pragma omp parallel for collapse(2)
    for (int r = 0; r < ROWS; r++)
        for (int c = 0; c < COLS; c++)
            apply_smoothing(r, c);
}

typedef struct { int start_row, end_row; } thread_data_t;
void* smooth_thread_func(void* arg) {
    thread_data_t* data = (thread_data_t*)arg;
    for (int r = data->start_row; r < data->end_row; r++)
        for (int c = 0; c < COLS; c++)
            apply_smoothing(r, c);
    return NULL;
}

void smooth_the_pthreads(int num_threads) {
    pthread_t threads[num_threads];
    thread_data_t thread_data[num_threads];
    int rpt = ROWS / num_threads;
    for (int i = 0; i < num_threads; i++) {
        thread_data[i].start_row = i * rpt;
        thread_data[i].end_row = (i == num_threads - 1) ? ROWS : (i + 1) * rpt;
        pthread_create(&threads[i], NULL, smooth_thread_func, &thread_data[i]);
    }
    for (int i = 0; i < num_threads; i++) pthread_join(threads[i], NULL);
}

double get_time() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

int main() {
    load_grid("grid_z.csv");
    double start, seq_time;
    int t_counts[] = {1, 2, 4, 8, 12, 16, 24};
    int num_tests = sizeof(t_counts) / sizeof(t_counts[0]);
    double omp_times[num_tests];
    double pth_times[num_tests];

    // 1. Run Sequential Baseline
    start = get_time();
    for(int i=0; i<ITERATIONS; i++) smooth_in_sequential();
    seq_time = get_time() - start;

    // 2. Run OpenMP Tests
    for(int i=0; i < num_tests; i++) {
        start = get_time();
        for(int j=0; j<ITERATIONS; j++) smooth_openmp(t_counts[i]);
        omp_times[i] = get_time() - start;
    }

    // 3. Run pThreads Tests
    for(int i=0; i < num_tests; i++) {
        if (t_counts[i] > ROWS) {
            pth_times[i] = -1.0; // Mark as skipped
            continue;
        }
        start = get_time();
        for(int j=0; j<ITERATIONS; j++) smooth_the_pthreads(t_counts[i]);
        pth_times[i] = get_time() - start;
    }

    // 4. Print Comparison Table
    printf("\n=== Performance Comparison (Problem Size: 24x200, 5000 Iterations) ===\n");
    printf("Sequential Baseline Time: %.6f seconds\n\n", seq_time);

    printf("%-8s | %-15s | %-15s | %-10s | %-10s\n", 
           "Threads", "OpenMP (s)", "pThreads (s)", "Speedup OMP", "Speedup PTH");
    printf("--------------------------------------------------------------------------\n");

    for(int i=0; i < num_tests; i++) {
        printf("%-8d | %-15.6f | ", t_counts[i], omp_times[i]);
        
        if (pth_times[i] > 0) {
            printf("%-15.6f | ", pth_times[i]);
            printf("%-11.2fx | %-10.2fx\n", seq_time / omp_times[i], seq_time / pth_times[i]);
        } else {
            printf("%-15s | %-11.2fx | %-10s\n", "N/A (T > Rows)", seq_time / omp_times[i], "N/A");
        }
    }
    printf("--------------------------------------------------------------------------\n\n");

    return 0;
}
