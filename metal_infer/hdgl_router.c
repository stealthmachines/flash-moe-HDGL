// hdgl_router.c — HDGL recursive temporal routing
// Part of the HDGL-28 Hypervisor-MoE bolster

#include <stdint.h>
#include <stdlib.h>
#include "bootloaderZ.h"
#include "slot4096.h"

// External state defined in the host inference engine
extern int num_experts;
extern int lattice_size;

// ---------------------------
// Shader Dimension Guard
// Pack these to pass into buffer(3) in shaders.metal to prevent out-of-bounds 
// reads on non-square (Up/Down projection) MoE layers.
typedef struct {
    uint32_t in_dim;
    uint32_t out_dim;
} HDGL_ShaderDims;

// ---------------------------
// Token Definition
typedef struct {
    char *text;
    int id;
} Token;

// ---------------------------
// History Vector (Temporal Context)
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
// FNV-1a Hash for Token Encoding
static uint32_t hash_token(Token t) {
    uint32_t hash = 2166136261u;
    for (char *p = t.text; p && *p; ++p) {
        hash ^= (uint32_t)(*p);
        hash *= 16777619u;
    }
    return hash;
}

// ---------------------------
// Encode Token → Slot4096 (APA Entry)
Slot4096 encode_token(Token t) {
    uint32_t h = hash_token(t);
    // Initialize 64-bit mantissa for the initial hash projection
    Slot4096 s = slot_init_apa(64, 16);
    s.mantissa_words[0] = ((uint64_t)h << 32) | h;
    s.exponent = 0;
    
    // Synchronize legacy exponent with the MPI backend
    mpi_update_from_legacy(&s.exponent_mpi, s.exponent);
    return s;
}

// ---------------------------
// Project Slot → Lattice Coordinate
int project_to_lattice(Slot4096 s) {
    // Determine the node index within the HDGL lattice
    return (int)(s.mantissa_words[0] % lattice_size);
}

// ---------------------------
// Query lattice for dynamic feedback
uint32_t lattice_feedback(Slot4096 s) {
    int lattice_id = project_to_lattice(s);
    HDGL_SlotState *state = hdgl_get_slot_state(lattice_id);
    if (!state) return 0;

    // Combine lattice state fields for recursive feedback
    // Represents the "vibrational state" of the lattice node
    uint32_t feedback = (uint32_t)(
        state->charge   ^
        state->entropy  ^
        state->tension
    );
    return feedback;
}

// ---------------------------
// Recursive routing using history
// Ties the current token route to the state of the Hypervisor history
int route_token_recursive(Token t, HDGL_History *H) {
    Slot4096 s = encode_token(t);

    uint32_t feedback = lattice_feedback(s);

    // Incorporate history: Current path = f(Lattice, Prev_Feedback, Prev_Expert)
    uint32_t combined = feedback ^ H->last_feedback ^ (uint32_t)H->last_expert_id;

    int lattice_id = project_to_lattice(s);
    int expert_id = (lattice_id + combined) % num_experts;

    // Update history for temporal continuity
    H->last_feedback = feedback;
    H->last_expert_id = expert_id;

    // Clean up APA slot to prevent memory leaks in the inference loop
    ap_free(&s);
    return expert_id;
}

// ---------------------------
// Batch routing with recursion
// Used by the inference server to process token sequences
void route_tokens_recursive(Token *tokens, int *expert_ids, int batch_size) {
    HDGL_History H = {0, 0}; // Initialize local temporal history for this sequence
    for (int i = 0; i < batch_size; ++i) {
        expert_ids[i] = route_token_recursive(tokens[i], &H);
    }
}

// ---------------------------
// Shader Utility: Dim Preparation
// Called by the Metal Command Encoder to populate buffer(3)
HDGL_ShaderDims hdgl_get_packed_dims(int in_dim, int out_dim) {
    HDGL_ShaderDims d;
    d.in_dim = (uint32_t)in_dim;
    d.out_dim = (uint32_t)out_dim;
    return d;
}
