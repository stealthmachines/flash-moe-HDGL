// hdgl_router.c ? HDGL-28 recursive temporal token router
// Corrected from flash-moe-HDGL fork:
//   - Replaced #include "bootloaderZ.h"/"slot4096.h" with "hdgl_bootloaderz.h"
//   - Replaced mpi_update_from_legacy() with inline from hdgl_bootloaderz.h
//   - Replaced extern num_experts/lattice_size with module-local state set by hdgl_router_init()
//   - lattice_get_slot() now performs bounds-checked total-slot lookup (not raw lattice_size)
//   - encode_token: slot_init_apa(64, 16) ? correct (1 mantissa word for token hash)
//   - hdgl_get_slot_state() lives in hdgl_bootloaderz.c and uses g_hdgl_lattice
#include "hdgl_router.h"
#include <stdint.h>
#include <stdlib.h>

// ?? Module state (set by hdgl_router_init) ???????????????????????????????????
static HDGLLattice *s_lattice    = NULL;
static int          s_num_experts = 0;

void hdgl_router_init(HDGLLattice *lat, int num_experts) {
    s_lattice     = lat;
    s_num_experts = num_experts;
}

// ?? FNV-1a hash for token text ???????????????????????????????????????????????
static uint32_t hash_token(Token t) {
    uint32_t hash = 2166136261u;
    for (char *p = t.text; p && *p; ++p) {
        hash ^= (uint32_t)(*p);
        hash *= 16777619u;
    }
    return hash;
}

// ?? Encode token ? Slot4096 (single-word APA entry for hash projection) ??????
static Slot4096 encode_token(Token t) {
    uint32_t h = hash_token(t);
    // bits_mant=64 ? num_words=1; sufficient for a 64-bit token hash seed
    Slot4096 s = slot_init_apa(64, 16);
    s.mantissa_words[0] = ((uint64_t)h << 32) | h;
    s.exponent = 0;
    mpi_update_from_legacy(&s.exponent_mpi, s.exponent);
    return s;
}

// ?? Project slot mantissa ? lattice slot index ???????????????????????????????
static int project_to_lattice(Slot4096 s) {
    if (!s_lattice) return 0;
    int total = s_lattice->num_instances * s_lattice->slots_per_instance;
    if (total <= 0) return 0;
    return (int)(s.mantissa_words[0] % (uint64_t)total);
}

// ?? Query lattice for dynamic feedback from slot state ???????????????????????
static uint32_t lattice_feedback(Slot4096 s) {
    int lattice_id = project_to_lattice(s);
    HDGL_SlotState *state = hdgl_get_slot_state(lattice_id);
    if (!state) return 0;
    return state->charge ^ state->entropy ^ state->tension;
}

// ?? Recursive routing: combines lattice state with temporal history ???????????
int route_token_recursive(Token t, HDGL_History *H) {
    if (s_num_experts <= 0) return 0;
    Slot4096 s = encode_token(t);
    uint32_t feedback = lattice_feedback(s);
    uint32_t combined = feedback ^ H->last_feedback ^ (uint32_t)H->last_expert_id;
    int lattice_id = project_to_lattice(s);
    int expert_id  = (int)((lattice_id + (int)combined) % s_num_experts);
    if (expert_id < 0) expert_id += s_num_experts;
    H->last_feedback  = feedback;
    H->last_expert_id = expert_id;
    ap_free(&s);
    return expert_id;
}

// ?? Batch routing ?????????????????????????????????????????????????????????????
void route_tokens_recursive(Token *tokens, int *expert_ids, int batch_size) {
    HDGL_History H = {0, 0};
    for (int i = 0; i < batch_size; ++i)
        expert_ids[i] = route_token_recursive(tokens[i], &H);
}

// ?? Shader utility ????????????????????????????????????????????????????????????
HDGL_ShaderDims hdgl_get_packed_dims(int in_dim, int out_dim) {
    HDGL_ShaderDims d;
    d.in_dim  = (uint32_t)in_dim;
    d.out_dim = (uint32_t)out_dim;
    return d;
}
