// hdgl_router.c — HDGL recursive temporal routing

#include <stdint.h>
#include <stdlib.h>
#include "bootloaderZ.h"
#include "slot4096.h"

extern int num_experts;
extern int lattice_size;

// ---------------------------
// Token Definition
typedef struct {
    char *text;
    int id;
} Token;

// ---------------------------
// History Vector
typedef struct {
    uint32_t last_feedback;
    int last_expert_id;
} HDGL_History;

// ---------------------------
// Helpers
Slot4096 encode_token(Token t);
int project_to_lattice(Slot4096 s);
uint32_t lattice_feedback(Slot4096 s);
int route_token_recursive(Token t, HDGL_History *H);

// ---------------------------
// Hash / Encoding
static uint32_t hash_token(Token t) {
    uint32_t hash = 2166136261u;
    for (char *p = t.text; p && *p; ++p) {
        hash ^= (uint32_t)(*p);
        hash *= 16777619u;
    }
    return hash;
}

// ---------------------------
// Encode Token → Slot4096
Slot4096 encode_token(Token t) {
    uint32_t h = hash_token(t);
    Slot4096 s = slot_init_apa(64, 16);
    s.mantissa_words[0] = ((uint64_t)h << 32) | h;
    s.exponent = 0;
    mpi_update_from_legacy(&s.exponent_mpi, s.exponent);
    return s;
}

// ---------------------------
// Project Slot → Lattice
int project_to_lattice(Slot4096 s) {
    return (int)(s.mantissa_words[0] % lattice_size);
}

// ---------------------------
// Query lattice for dynamic feedback
uint32_t lattice_feedback(Slot4096 s) {
    int lattice_id = project_to_lattice(s);
    HDGL_SlotState *state = hdgl_get_slot_state(lattice_id);
    if (!state) return 0;

    // combine lattice fields for feedback
    uint32_t feedback = (uint32_t)(
        state->charge   ^
        state->entropy  ^
        state->tension
    );
    return feedback;
}

// ---------------------------
// Recursive routing using history
int route_token_recursive(Token t, HDGL_History *H) {
    Slot4096 s = encode_token(t);

    uint32_t feedback = lattice_feedback(s);

    // Incorporate history
    uint32_t combined = feedback ^ H->last_feedback ^ (uint32_t)H->last_expert_id;

    int lattice_id = project_to_lattice(s);
    int expert_id = (lattice_id + combined) % num_experts;

    // Update history
    H->last_feedback = feedback;
    H->last_expert_id = expert_id;

    ap_free(&s);
    return expert_id;
}

// ---------------------------
// Batch routing with recursion
void route_tokens_recursive(Token *tokens, int *expert_ids, int batch_size) {
    HDGL_History H = {0, 0}; // initialize history
    for (int i = 0; i < batch_size; ++i) {
        expert_ids[i] = route_token_recursive(tokens[i], &H);
    }
}
