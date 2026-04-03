// hdgl_lattice_generator.c - HDGL-28 v2.0 lattice pre-seeder
//
// Generates hdgl_lattice.bin: a pre-seeded lattice state file that can be
// loaded by infer/chat instead of re-seeding from scratch at startup.

#include "hdgl_bootloaderz.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>

#define MAGIC    "HDGL"
#define VERSION  0x00020000u

typedef struct {
    uint64_t mantissa_word0;
    int64_t  exponent;
    double   phase;
    double   freq;
    uint32_t state_flags;
    uint32_t strand_idx;
} SlotRecord;

static void write_u32(FILE *f, uint32_t v) { fwrite(&v, 4, 1, f); }
static void write_f64(FILE *f, double   v) { fwrite(&v, 8, 1, f); }

int main(int argc, char **argv) {
    int    num_instances   = 4096;
    int    slots_per       = BLZ_SLOTS_PER_INST;
    int    steps           = 200;
    const char *outfile    = "hdgl_lattice.bin";
    unsigned long seed     = (unsigned long)time(NULL);

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--instances")   && i+1 < argc) { num_instances = atoi(argv[++i]); }
        else if (!strcmp(argv[i], "--slots-per") && i+1 < argc) { slots_per = atoi(argv[++i]); }
        else if (!strcmp(argv[i], "--steps")   && i+1 < argc) { steps = atoi(argv[++i]); }
        else if (!strcmp(argv[i], "--output")  && i+1 < argc) { outfile = argv[++i]; }
        else if (!strcmp(argv[i], "--seed")    && i+1 < argc) { seed = (unsigned long)atoi(argv[++i]); }
        else { fprintf(stderr, "Unknown option: %s\n", argv[i]); return 1; }
    }

    srand((unsigned int)seed);
    printf("[hdgl_lattice_generator] v%s\n", HDGL_VERSION_STR);
    printf("  instances=%d  slots_per=%d  steps=%d  seed=%lu\n",
           num_instances, slots_per, steps, seed);
    printf("  output=%s\n", outfile);

    HDGLLattice *lat = lattice_init(num_instances, slots_per);
    if (!lat) { fprintf(stderr, "ERROR: lattice_init failed.\n"); return 1; }
    g_hdgl_lattice = lat;

    bootloader_init_lattice(lat, steps);

    FILE *f = fopen(outfile, "wb");
    if (!f) { fprintf(stderr, "ERROR: cannot open %s\n", outfile); return 1; }

    fwrite(MAGIC, 1, 4, f);
    write_u32(f, VERSION);
    write_u32(f, (uint32_t)num_instances);
    write_u32(f, (uint32_t)slots_per);
    write_f64(f, lat->time);
    write_f64(f, lat->omega);
    write_f64(f, lat->phase_var);

    int total = num_instances * slots_per;
    int written = 0, skipped = 0;

    for (int i = 0; i < total; i++) {
        Slot4096 *s = lattice_get_slot(lat, i);
        SlotRecord rec = {0};
        if (s && s->mantissa_words) {
            rec.mantissa_word0 = s->mantissa_words[0];
            rec.exponent       = s->exponent;
            rec.phase          = s->phase;
            rec.freq           = s->freq;
            rec.state_flags    = s->state_flags;
            rec.strand_idx     = (uint32_t)(i % SPIRAL8_GEOMETRIES);
            written++;
        } else {
            skipped++;
        }
        fwrite(&rec, sizeof(rec), 1, f);
    }

    fclose(f);

    printf("[hdgl_lattice_generator] Done: %d slots written, %d skipped\n", written, skipped);
    printf("  File: %s  (%.1f MB)\n", outfile,
           (double)(total * sizeof(SlotRecord) + 36) / (1024.0*1024.0));

    g_hdgl_lattice = NULL;
    lattice_free(lat);
    return 0;
}
