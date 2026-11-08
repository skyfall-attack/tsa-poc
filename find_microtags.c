#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/mman.h>
#include <time.h>
#include <x86intrin.h>

#define WARMUP_ITERATIONS 100
#define MEASURE_ITERATIONS 5000
#define TRIALS 10
// For Zen 5 the threshhold is around 4 cycles
#define THRESHOLD 15.0

#define BASE_VA 0x200000000000ULL
#define MAX_BIT 47

typedef struct {
    double min;
    double max;
    double avg;
} LatencyStats;

double get_time() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec / 1e9;
}

void* map_target(uint64_t offset) {
    void* ptr = mmap((void*)(BASE_VA + offset), 4096, PROT_READ | PROT_WRITE,
                     MAP_PRIVATE | MAP_ANONYMOUS | MAP_FIXED, -1, 0);
    if (ptr != MAP_FAILED) *(volatile char*)ptr = 1;
    return ptr;
}

LatencyStats measure_latency(void** ptr1, void** ptr2) {
    *ptr1 = ptr2;
    *ptr2 = ptr1;
    unsigned int aux;

    double min_lat = 100000.0, max_lat = 0.0, sum_lat = 0.0;

    for (int t = 0; t < TRIALS; t++) {
        void** p = ptr1;
        for (int i = 0; i < WARMUP_ITERATIONS; i++) p = (void**)*p;

        uint64_t start = __rdtscp(&aux);
        for (int i = 0; i < MEASURE_ITERATIONS; i++) {
            p = (void**)*p;
            p = (void**)*p;
            p = (void**)*p;
            p = (void**)*p;
            p = (void**)*p;
            p = (void**)*p;
            p = (void**)*p;
            p = (void**)*p;
            p = (void**)*p;
            p = (void**)*p;
        }
        uint64_t end = __rdtscp(&aux);

        if (!p) printf("Dummy\n");

        double current_latency = (double)(end - start) / (MEASURE_ITERATIONS * 10.0);

        if (current_latency < min_lat) min_lat = current_latency;
        if (current_latency > max_lat) max_lat = current_latency;
        sum_lat += current_latency;
    }
    return (LatencyStats){min_lat, max_lat, sum_lat / TRIALS};
}

void print_equation(int index, uint64_t vector) {
    printf("  uint64_t u%d = ", index);
    bool first = true;
    for (int i = 0; i <= MAX_BIT; i++) {
        if ((vector >> i) & 1) {
            if (!first) printf(" ^ ");
            printf("GET_BIT(va, %2d)", i);
            first = false;
        }
    }
    printf(";\n");
}

// Dynamically calculates the utag using the discovered basis vectors
uint64_t compute_dynamic_utag(uint64_t va, uint64_t* basis, int basis_count) {
    uint64_t utag = 0;
    for (int i = 0; i < basis_count; i++) {
        // __builtin_popcountll(va & basis[i]) % 2 is equivalent to multiple XORs
        uint64_t bit = __builtin_popcountll(va & basis[i]) % 2;
        utag |= (bit << i);
    }
    return utag;
}

int main() {
    double total_start = get_time();
    printf("[+] Establishing Base Virtual Address at 0x%llX...\n", BASE_VA);
    void* base_ptr = map_target(0);
    if (base_ptr == MAP_FAILED) {
        perror("Base mmap failed");
        return 1;
    }

    uint64_t active_bits_mask = 0;

    // --- PHASE 1 ---
    printf("\n[+] Phase 1: Scanning 128TB VA Space (Bits 12 to %d)\n", MAX_BIT);
    for (int i = 12; i <= MAX_BIT; i++) {
        uint64_t offset = 1ULL << i;
        void* target_ptr = map_target(offset);
        if (target_ptr == MAP_FAILED) continue;

        LatencyStats stats = measure_latency((void**)base_ptr, (void**)target_ptr);
        munmap(target_ptr, 4096);

        if (stats.min <= THRESHOLD) active_bits_mask |= (1ULL << i);
    }
    printf("    Active bits mask: 0x%016llX\n", active_bits_mask);
    if (active_bits_mask == 0) return 1;

    // --- PHASE 2 ---
    printf("[+] Phase 2: Probing Active Bit Combinations\n");
    uint64_t collisions[1024];
    int col_count = 0;

    for (int i = 12; i <= MAX_BIT; i++) {
        if (!((active_bits_mask >> i) & 1)) continue;
        for (int j = i + 1; j <= MAX_BIT; j++) {
            if (!((active_bits_mask >> j) & 1)) continue;
            uint64_t offset = (1ULL << i) | (1ULL << j);
            void* target_ptr = map_target(offset);
            if (measure_latency((void**)base_ptr, (void**)target_ptr).min > THRESHOLD) {
                collisions[col_count++] = offset;
            }
            munmap(target_ptr, 4096);
        }
    }
    for (int i = 12; i <= MAX_BIT; i++) {
        if (!((active_bits_mask >> i) & 1)) continue;
        for (int j = i + 1; j <= MAX_BIT; j++) {
            if (!((active_bits_mask >> j) & 1)) continue;
            for (int k = j + 1; k <= MAX_BIT; k++) {
                if (!((active_bits_mask >> k) & 1)) continue;
                uint64_t offset = (1ULL << i) | (1ULL << j) | (1ULL << k);
                void* target_ptr = map_target(offset);
                if (measure_latency((void**)base_ptr, (void**)target_ptr).min > THRESHOLD) {
                    collisions[col_count++] = offset;
                }
                munmap(target_ptr, 4096);
            }
        }
    }
    printf("    Found %d independent collision vectors.\n", col_count);

    // --- PHASE 3 ---
    printf("[+] Phase 3: Solving 64-Bit GF(2) Orthogonal Matrix\n");
    uint64_t basis[64];
    int basis_count = 0;

    uint64_t v = active_bits_mask;
    do {
        bool valid = true;
        for (int i = 0; i < col_count; i++) {
            if (__builtin_popcountll(v & collisions[i]) % 2 != 0) {
                valid = false;
                break;
            }
        }
        if (valid) {
            uint64_t temp = v;
            for (int i = 0; i < basis_count; i++) {
                int msb = 63 - __builtin_clzll(basis[i]);
                if ((temp >> msb) & 1) temp ^= basis[i];
            }
            if (temp != 0) {
                basis[basis_count++] = temp;
                for (int i = basis_count - 1; i > 0; i--) {
                    if (basis[i] > basis[i - 1]) {
                        uint64_t swap = basis[i];
                        basis[i] = basis[i - 1];
                        basis[i - 1] = swap;
                    }
                }
            }
        }
        v = (v - 1) & active_bits_mask;
    } while (v != 0);

    // Print the equations
    printf("\n==== FINAL HARDWARE EQUATIONS ====\n");
    printf("#define GET_BIT(va, bit) (((uint64_t)(va) >> (bit)) & 1ULL)\n\n");
    printf("uint64_t calculate_zen_utag(void *ptr) {\n");
    printf("  uint64_t va = (uint64_t)ptr;\n");
    for (int i = 0; i < basis_count; i++) print_equation(i, basis[i]);
    printf("  \n  return ");
    for (int i = 0; i < basis_count; i++) {
        if (i > 0) printf(" | ");
        printf("(u%d << %d)", i, i);
    }
    printf(";\n}\n");
    printf("==================================\n");

    // --- PHASE 4: THE EMPIRICAL VALIDATION ---
    printf("\n[+] Phase 4: Validating Equations in Real-World Environment\n");

    // Allocate two standard 64MB buffers, just like a real program would
    size_t val_size = 1024 * 1024 * 64;
    char* bufA = mmap(NULL, val_size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    char* bufB = mmap(NULL, val_size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);

    if (bufA == MAP_FAILED || bufB == MAP_FAILED) {
        perror("Validation mmap failed");
        return 1;
    }

    // Page fault them so physical memory is backed
    bufA[0] = 1;
    bufB[0] = 1;

    uint64_t va_A = (uint64_t)bufA;
    uint64_t utag_A = compute_dynamic_utag(va_A, basis, basis_count);
    printf("    Buffer A allocated at: 0x%016llX (Computed UTag: 0x%llX)\n", va_A, utag_A);
    printf("    Buffer B allocated at: 0x%016llX\n", (uint64_t)bufB);

    uint64_t offset_collide = -1;
    uint64_t offset_safe = -1;

    // Scan Buffer B for offsets that match/mismatch the UTag
    // We step by 4096 (page size) to guarantee the 12-bit cache set index remains identical
    for (uint64_t off = 0; off < val_size; off += 4096) {
        uint64_t va_B = (uint64_t)(bufB + off);
        uint64_t utag_B = compute_dynamic_utag(va_B, basis, basis_count);

        if (utag_A == utag_B && offset_collide == (uint64_t)-1) offset_collide = off;
        if (utag_A != utag_B && offset_safe == (uint64_t)-1) offset_safe = off;

        if (offset_collide != (uint64_t)-1 && offset_safe != (uint64_t)-1) break;
    }

    printf("\n    --> Testing Predicted Collision...\n");
    printf("        Address: 0x%016llX (Computed UTag: 0x%llX)\n", (uint64_t)(bufB + offset_collide), compute_dynamic_utag((uint64_t)(bufB + offset_collide), basis, basis_count));
    LatencyStats stat_col = measure_latency((void**)bufA, (void**)(bufB + offset_collide));
    printf("        Result: Min: %5.2f | Max: %5.2f | Avg: %5.2f cycles\n", stat_col.min, stat_col.max, stat_col.avg);
    if (stat_col.min > THRESHOLD)
        printf("        [SUCCESS] Equation accurately predicted L1D Thrashing.\n");
    else
        printf("        [FAILED] Expected high latency, got low latency.\n");

    printf("\n    --> Testing Predicted Safe Misalignment...\n");
    printf("        Address: 0x%016llX (Computed UTag: 0x%llX)\n", (uint64_t)(bufB + offset_safe), compute_dynamic_utag((uint64_t)(bufB + offset_safe), basis, basis_count));
    LatencyStats stat_safe = measure_latency((void**)bufA, (void**)(bufB + offset_safe));
    printf("        Result: Min: %5.2f | Max: %5.2f | Avg: %5.2f cycles\n", stat_safe.min, stat_safe.max, stat_safe.avg);
    if (stat_safe.min < THRESHOLD)
        printf("        [SUCCESS] Equation accurately predicted L1D Hit.\n");
    else
        printf("        [FAILED] Expected low latency, got high latency.\n");

    munmap(bufA, val_size);
    munmap(bufB, val_size);
    munmap(base_ptr, 4096);
    return 0;
}