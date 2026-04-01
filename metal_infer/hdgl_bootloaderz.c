// hdgl_bootloaderz.c ? BootloaderZ V6.0 APA/HDGL lattice library
// Extracted from bootloaderZ.zip (bootloaderZ.c), main() removed.
// Integrated into Flash-MoE as the HDGL-28 hyper-precision backend.
// Bug fixes applied: ap_copy aliased UAF, calloc for chunk slots, B_aligned zero-init.
#include "hdgl_bootloaderz.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#define PHI           BLZ_PHI
#define MAX_INSTANCES BLZ_MAX_INSTANCES
#define CHUNK_SIZE    BLZ_CHUNK_SIZE
#define MSB_MASK      BLZ_MSB_MASK
#define MPI_INITIAL_WORDS 1

static const float fib_table[]  = {1,1,2,3,5,8,13,21,34,55,89,144,233,377,610,987};
static const float prime_table[] = {2,3,5,7,11,13,17,19,23,29,31,37,41,43,47,53};
static const int   fib_len   = 16;
static const int   prime_len = 16;

static double get_normalized_rand(void) { return (double)rand() / RAND_MAX; }
#define GET_RANDOM_UINT64() (((uint64_t)rand() << 32) | (uint64_t)rand())

// ?? Global high-precision constants ?????????????????????????????????????????
static Slot4096 APA_CONST_PHI;
static Slot4096 APA_CONST_PI;
static int       s_apa_constants_initialized = 0;

// ?? Global lattice pointer (extern'd by hdgl_router.c) ???????????????????????
HDGLLattice *g_hdgl_lattice = NULL;

// ?? Forward declarations ?????????????????????????????????????????????????????
static void ap_normalize_legacy(Slot4096 *slot);
static void ap_add_legacy(Slot4096 *A, const Slot4096 *B);
static void ap_shift_right_legacy(uint64_t *mantissa_words, size_t num_words, int64_t shift_amount);

// ?? MPI ??????????????????????????????????????????????????????????????????????
void mpi_init(MPI *m, size_t initial_words) {
    m->words = calloc(initial_words, sizeof(uint64_t));
    m->num_words = initial_words;
    m->sign = 0;
}
void mpi_free(MPI *m) {
    if (m->words) free(m->words);
    m->words = NULL;
    m->num_words = 0;
}
void mpi_copy(MPI *dest, const MPI *src) {
    mpi_free(dest);
    dest->num_words = src->num_words;
    dest->words = malloc(src->num_words * sizeof(uint64_t));
    if (src->words && dest->words)
        memcpy(dest->words, src->words, src->num_words * sizeof(uint64_t));
    dest->sign = src->sign;
}
void mpi_set_value(MPI *m, uint64_t value, uint8_t sign) {
    if (m->words) m->words[0] = value;
    m->sign = sign;
}
static void mpi_resize(MPI *m, size_t nw)        { (void)m; (void)nw; }
static int  mpi_compare(const MPI *A, const MPI *B) { (void)A; (void)B; return 0; }
static void mpi_add_op(MPI *A, const MPI *B)     { (void)A; (void)B; }
static void mpi_subtract_op(MPI *A, const MPI *B){ (void)A; (void)B; }
static size_t mpi_get_effective_words(const MPI *m) { return m->num_words; }
static int    mpi_count_leading_zeros(const MPI *m) { (void)m; return 64; }

// ?? Slot4096 ?????????????????????????????????????????????????????????????????
Slot4096 slot_init_apa(int bits_mant, int bits_exp) {
    Slot4096 slot = {0};
    slot.bits_mant = bits_mant;
    slot.bits_exp  = bits_exp;
    slot.num_words = (size_t)((bits_mant + 63) / 64);
    slot.mantissa_words = calloc(slot.num_words, sizeof(uint64_t));
    mpi_init(&slot.exponent_mpi,       MPI_INITIAL_WORDS);
    mpi_init(&slot.num_words_mantissa, MPI_INITIAL_WORDS);
    mpi_init(&slot.source_of_infinity, MPI_INITIAL_WORDS);
    if (!slot.mantissa_words) { fprintf(stderr, "Error: mantissa alloc failed.\n"); return slot; }
    if (slot.num_words > 0) { slot.mantissa_words[0] = GET_RANDOM_UINT64(); slot.mantissa_words[0] |= MSB_MASK; }
    int64_t exp_range = 1LL << bits_exp;
    int64_t exp_bias  = 1LL << (bits_exp - 1);
    slot.exponent = (rand() % exp_range) - exp_bias;
    slot.base     = (float)(PHI + get_normalized_rand() * 0.01);
    slot.exponent_base = 4096;
    mpi_set_value(&slot.exponent_mpi,       (uint64_t)llabs(slot.exponent), slot.exponent < 0 ? 1 : 0);
    mpi_set_value(&slot.num_words_mantissa, (uint64_t)slot.num_words,       0);
    return slot;
}

void ap_free(Slot4096 *slot) {
    if (!slot) return;
    if (slot->mantissa_words) { free(slot->mantissa_words); slot->mantissa_words = NULL; }
    mpi_free(&slot->exponent_mpi);
    mpi_free(&slot->num_words_mantissa);
    mpi_free(&slot->source_of_infinity);
    slot->num_words = 0;
}

// FIX: zero MPI fields after shallow copy to prevent mpi_copy freeing src's memory
void ap_copy(Slot4096 *dest, const Slot4096 *src) {
    ap_free(dest);
    *dest = *src;
    dest->exponent_mpi.words       = NULL; dest->exponent_mpi.num_words       = 0;
    dest->num_words_mantissa.words = NULL; dest->num_words_mantissa.num_words = 0;
    dest->source_of_infinity.words = NULL; dest->source_of_infinity.num_words = 0;
    dest->mantissa_words = malloc(src->num_words * sizeof(uint64_t));
    if (!dest->mantissa_words) { fprintf(stderr, "Error: deep copy alloc failed.\n"); dest->num_words = 0; return; }
    memcpy(dest->mantissa_words, src->mantissa_words, src->num_words * sizeof(uint64_t));
    mpi_copy(&dest->exponent_mpi,       &src->exponent_mpi);
    mpi_copy(&dest->num_words_mantissa, &src->num_words_mantissa);
    mpi_copy(&dest->source_of_infinity, &src->source_of_infinity);
}

double ap_to_double(const Slot4096 *slot) {
    if (!slot || slot->num_words == 0 || !slot->mantissa_words) return 0.0;
    return ((double)slot->mantissa_words[0] / (double)UINT64_MAX) * pow(2.0, (double)slot->exponent);
}

Slot4096 *ap_from_double(double value, int bits_mant, int bits_exp) {
    Slot4096 temp = slot_init_apa(bits_mant, bits_exp);
    Slot4096 *slot = malloc(sizeof(Slot4096));
    if (!slot) { ap_free(&temp); return NULL; }
    *slot = temp;
    if (value == 0.0) return slot;
    int exp_offset;
    double mant_val = frexp(value, &exp_offset);
    slot->mantissa_words[0] = (uint64_t)(mant_val * (double)UINT64_MAX);
    slot->exponent = (int64_t)exp_offset;
    mpi_set_value(&slot->exponent_mpi, (uint64_t)llabs(slot->exponent), slot->exponent < 0 ? 1 : 0);
    return slot;
}

static void ap_shift_right_legacy(uint64_t *mw, size_t nw, int64_t shift) {
    if (shift <= 0 || nw == 0) return;
    if (shift >= (int64_t)nw * 64) { memset(mw, 0, nw * sizeof(uint64_t)); return; }
    int64_t ws = shift / 64;
    int     bs = (int)(shift % 64);
    if (ws > 0) {
        for (size_t i = nw; i-- > (size_t)ws; ) mw[i] = mw[i - ws];
        memset(mw, 0, (size_t)ws * sizeof(uint64_t));
    }
    if (bs > 0) {
        int rs = 64 - bs;
        for (size_t i = nw; i-- > 0; ) {
            uint64_t carry = (i > 0) ? (mw[i-1] << rs) : 0;
            mw[i] = (mw[i] >> bs) | carry;
        }
    }
}
void ap_shift_right(uint64_t *mw, size_t nw, int64_t shift) { ap_shift_right_legacy(mw, nw, shift); }

static void ap_normalize_legacy(Slot4096 *slot) {
    if (slot->num_words == 0) return;
    while (!(slot->mantissa_words[0] & MSB_MASK)) {
        if (slot->exponent <= -(1LL << (slot->bits_exp - 1))) { slot->state_flags |= APA_FLAG_GUZ; break; }
        uint64_t carry = 0;
        for (size_t i = slot->num_words; i-- > 0; ) {
            uint64_t nc = (slot->mantissa_words[i] & MSB_MASK) ? 1 : 0;
            slot->mantissa_words[i] = (slot->mantissa_words[i] << 1) | carry;
            carry = nc;
        }
        slot->exponent--;
    }
    if (slot->mantissa_words[0] == 0) slot->exponent = 0;
}
void ap_normalize(Slot4096 *slot) { ap_normalize_legacy(slot); }

// FIX: B_aligned = {0} to prevent ap_copy calling ap_free on garbage stack pointer
static void ap_add_legacy(Slot4096 *A, const Slot4096 *B) {
    if (A->num_words != B->num_words) { fprintf(stderr, "Error: APA add word count mismatch.\n"); return; }
    Slot4096 B_aligned = {0};
    ap_copy(&B_aligned, B);
    int64_t ed = A->exponent - B_aligned.exponent;
    if      (ed > 0) { ap_shift_right(B_aligned.mantissa_words, B_aligned.num_words,  ed); B_aligned.exponent = A->exponent; }
    else if (ed < 0) { ap_shift_right(A->mantissa_words,        A->num_words,         -ed); A->exponent = B_aligned.exponent; }
    uint64_t carry = 0;
    size_t nw = A->num_words;
    for (size_t i = nw; i-- > 0; ) {
        uint64_t s = A->mantissa_words[i] + B_aligned.mantissa_words[i] + carry;
        carry = (s < A->mantissa_words[i] || (s == A->mantissa_words[i] && carry)) ? 1 : 0;
        A->mantissa_words[i] = s;
    }
    if (carry) {
        if (A->exponent >= (1LL << (A->bits_exp - 1))) { A->state_flags |= APA_FLAG_GOI; }
        else { A->exponent++; ap_shift_right(A->mantissa_words, nw, 1); A->mantissa_words[0] |= MSB_MASK; }
    }
    ap_normalize(A);
    mpi_set_value(&A->exponent_mpi, (uint64_t)llabs(A->exponent), A->exponent < 0 ? 1 : 0);
    ap_free(&B_aligned);
}
void ap_add(Slot4096 *A, const Slot4096 *B) { ap_add_legacy(A, B); }

// ?? HDGL Lattice ?????????????????????????????????????????????????????????????
HDGLLattice *lattice_init(int num_instances, int slots_per_instance) {
    HDGLLattice *lat = malloc(sizeof(HDGLLattice));
    if (!lat) return NULL;
    lat->num_instances = num_instances; lat->slots_per_instance = slots_per_instance;
    lat->omega = 0.0; lat->time = 0.0;
    int total = num_instances * slots_per_instance;
    lat->num_chunks = (total + CHUNK_SIZE - 1) / CHUNK_SIZE;
    lat->chunks = calloc((size_t)lat->num_chunks, sizeof(HDGLChunk *));
    if (!lat->chunks) { free(lat); return NULL; }
    return lat;
}

HDGLChunk *lattice_get_chunk(HDGLLattice *lat, int idx) {
    if (idx >= lat->num_chunks) return NULL;
    if (!lat->chunks[idx]) {
        HDGLChunk *ch = malloc(sizeof(HDGLChunk));
        if (!ch) return NULL;
        ch->allocated = CHUNK_SIZE;
        // FIX: calloc so uninitialized slots have NULL mantissa_words ? safe for ap_free
        ch->slots = calloc(CHUNK_SIZE, sizeof(Slot4096));
        if (!ch->slots) { free(ch); return NULL; }
        for (int i = 0; i < CHUNK_SIZE; i++) {
            int bm = 4096 + (i % 8) * 64, be = 16 + (i % 8) * 2;
            ch->slots[i] = slot_init_apa(bm, be);
        }
        lat->chunks[idx] = ch;
    }
    return lat->chunks[idx];
}

Slot4096 *lattice_get_slot(HDGLLattice *lat, int idx) {
    if (!lat || idx < 0) return NULL;
    int total = lat->num_instances * lat->slots_per_instance;
    if (idx >= total) return NULL;
    int ci = idx / CHUNK_SIZE, li = idx % CHUNK_SIZE;
    HDGLChunk *ch = lattice_get_chunk(lat, ci);
    return ch ? &ch->slots[li] : NULL;
}

double prismatic_recursion(HDGLLattice *lat, int idx, double val) {
    double ph = pow(PHI, (double)(idx % 16)), fh = fib_table[idx % fib_len];
    double dy = (double)(1 << (idx % 16)), pr = prime_table[idx % prime_len];
    double om = 0.5 + 0.5 * sin(lat->time + idx * 0.01);
    double rd = pow(val, (double)((idx % 7) + 1));
    return sqrt(ph * fh * dy * pr * om) * rd;
}

void lattice_step_cpu(HDGLLattice *lat, double tick) {
    int total = lat->num_instances * lat->slots_per_instance;
    for (int i = 0; i < total; i++) {
        Slot4096 *s = lattice_get_slot(lat, i);
        if (!s || (s->state_flags & (APA_FLAG_GOI | APA_FLAG_IS_NAN))) continue;
        double val = ap_to_double(s), r = prismatic_recursion(lat, i, val);
        double inc_v = pow((double)s->base, (double)s->exponent) * tick + 0.05 * r;
        Slot4096 *inc = ap_from_double(inc_v, s->bits_mant, s->bits_exp);
        if (!inc) continue;
        ap_add(s, inc); ap_free(inc); free(inc);
    }
    lat->omega += 0.01 * tick; lat->time += tick;
}

void lattice_fold(HDGLLattice *lat) {
    int oi = lat->num_instances, ni = oi * 2;
    if (ni > MAX_INSTANCES) return;
    int ot = oi * lat->slots_per_instance, nt = ni * lat->slots_per_instance;
    int oc = lat->num_chunks, nc = (nt + CHUNK_SIZE - 1) / CHUNK_SIZE;
    HDGLChunk **np = realloc(lat->chunks, (size_t)nc * sizeof(HDGLChunk *));
    if (!np) { fprintf(stderr, "Failed to alloc for folding\n"); return; }
    lat->chunks = np;
    for (int i = oc; i < nc; i++) lat->chunks[i] = NULL;
    for (int i = 0; i < ot; i++) {
        Slot4096 *os = lattice_get_slot(lat, i), *ns = lattice_get_slot(lat, ot + i);
        if (os && ns) {
            ap_copy(ns, os);
            Slot4096 *p = ap_from_double(fib_table[i % fib_len] * 0.01, ns->bits_mant, ns->bits_exp);
            if (p) { ap_add(ns, p); ap_free(p); free(p); }
            ns->base += (float)(get_normalized_rand() * 0.001);
        }
    }
    lat->num_instances = ni; lat->num_chunks = nc;
}

void lattice_free(HDGLLattice *lat) {
    if (!lat) return;
    free_apa_constants();
    if (lat == g_hdgl_lattice) g_hdgl_lattice = NULL;
    for (int i = 0; i < lat->num_chunks; i++) {
        if (lat->chunks[i]) {
            for (size_t j = 0; j < CHUNK_SIZE; j++) ap_free(&lat->chunks[i]->slots[j]);
            free(lat->chunks[i]->slots); free(lat->chunks[i]);
        }
    }
    free(lat->chunks); free(lat);
}

// ?? Lattice slot state for router ????????????????????????????????????????????
HDGL_SlotState *hdgl_get_slot_state(int lattice_id) {
    static HDGL_SlotState state;
    Slot4096 *s = lattice_get_slot(g_hdgl_lattice, lattice_id);
    if (!s || !s->mantissa_words) return NULL;
    uint64_t w = s->mantissa_words[0];
    state.charge  = (uint32_t)(w & 0xFFFFFFFF);
    state.entropy = (uint32_t)(w >> 32);
    state.tension = (uint32_t)(s->exponent & 0xFFFFFFFF);
    return &state;
}

// ?? Bootloader integration ???????????????????????????????????????????????????
void init_apa_constants(void) {
    APA_CONST_PHI = slot_init_apa(4096, 16); APA_CONST_PI = slot_init_apa(4096, 16);
    Slot4096 *tp = ap_from_double(1.6180339887, APA_CONST_PHI.bits_mant, APA_CONST_PHI.bits_exp);
    ap_copy(&APA_CONST_PHI, tp); ap_free(tp); free(tp);
    Slot4096 *tpi = ap_from_double(3.1415926535, APA_CONST_PI.bits_mant, APA_CONST_PI.bits_exp);
    ap_copy(&APA_CONST_PI, tpi); ap_free(tpi); free(tpi);
    s_apa_constants_initialized = 1;
    printf("[Bootloader] High-precision constant slots (PHI, PI) initialized.\n");
}

void free_apa_constants(void) {
    if (!s_apa_constants_initialized) return;
    ap_free(&APA_CONST_PHI);
    ap_free(&APA_CONST_PI);
    s_apa_constants_initialized = 0;
}

void bootloader_init_lattice(HDGLLattice *lat, int steps) {
    printf("[Bootloader] Initializing HDGL lattice with FUTURE-PROOF APA V2.1...\n");
    if (!lat) { printf("[Bootloader] ERROR: Lattice allocation failed.\n"); return; }
    init_apa_constants();
    printf("[Bootloader] %d instances, %d total slots\n",
           lat->num_instances, lat->num_instances * lat->slots_per_instance);
    for (int i = 0; i < steps; i++) lattice_step_cpu(lat, 0.01);
    printf("[Bootloader] Lattice seeded with %d steps\n", steps);
    printf("[Bootloader] Omega: %.6f, Time: %.6f\n", lat->omega, lat->time);
}
