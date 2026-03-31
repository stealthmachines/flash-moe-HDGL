// hdgl_router.c — HDGL-aware routing

#include <stdint.h>
#include <stdlib.h>
#include "bootloaderZ.h"  // HDGL lattice primitives
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
// Forward Declarations
Slot4096 encode_token(Token t);
int project_to_lattice(Slot4096 s);
uint32_t lattice_feedback(Slot4096 s);  // NEW: query lattice
int route_token(Token t);

// ---------------------------
// Hash / Encoding Helpers
static uint32_t hash_token(Token t) {
    uint32_t hash = 2166136261u;
    for (char *p = t.text; p && *p; ++p) {
        hash ^= (uint32_t)(*p);
        hash *= 16777619u;
    }
    return hash;
}

// ---------------------------
// Token → Slot4096 Encoding
Slot4096 encode_token(Token t) {
    uint32_t h = hash_token(t);
    Slot4096 s = slot_init_apa(64, 16);
    s.mantissa_words[0] = ((uint64_t)h << 32) | h;
    s.exponent = 0;
    mpi_update_from_legacy(&s.exponent_mpi, s.exponent);
    return s;
}

// ---------------------------
// Slot → Lattice Projection
int project_to_lattice(Slot4096 s) {
    int lattice_id = (int)(s.mantissa_words[0] % lattice_size);
    return lattice_id;
}

// ---------------------------
// Query lattice for dynamic routing feedback
uint32_t lattice_feedback(Slot4096 s) {
    // Example: read Ω_n,b (tension/entropy) from lattice at projected slot
    int lattice_id = project_to_lattice(s);
    HDGL_SlotState *state = hdgl_get_slot_state(lattice_id); // bootloaderZ API
    if (!state) return 0;

    // Combine some lattice fields for routing influence
    uint32_t feedback = (uint32_t)(
        state->charge   ^   // Ω_n,b charge component
        state->entropy  ^   // entropy field
        state->tension      // field tension
    );

    return feedback;
}

// ---------------------------
// Main Routing Function
int route_token(Token t) {
    Slot4096 s = encode_token(t);

    // HDGL lattice-aware routing
    uint32_t feedback = lattice_feedback(s);
    int lattice_id = project_to_lattice(s);
    int expert_id = (lattice_id + feedback) % num_experts;

    ap_free(&s);  // clean up

    return expert_id;
}

// ---------------------------
// Optional batch routing
void route_tokens_batch(Token *tokens, int *expert_ids, int batch_size) {
    for (int i = 0; i < batch_size; ++i) {
        expert_ids[i] = route_token(tokens[i]);
    }
}
