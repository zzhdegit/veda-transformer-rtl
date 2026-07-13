`default_nettype none

package paper_score_packet_pkg;
    parameter int H9_META_W = 16;
    parameter int H9_HEAD_W = 8;
    parameter int H9_INDEX_W = 16;
    parameter int H9_STATUS_W = 8;

    typedef struct packed {
        logic [H9_META_W-1:0]   token_meta;
        logic [H9_HEAD_W-1:0]   head_id;
        logic [H9_INDEX_W-1:0]  logical_token_index;
        logic [H9_INDEX_W-1:0]  cache_slot;
        logic [H9_INDEX_W-1:0]  score_index;
        logic [H9_INDEX_W-1:0]  tile_id;
        logic [127:0]           lane_mask;
        logic [31:0]            score_fp32;
        logic                   last_in_tile;
        logic                   last_in_head;
        logic [H9_STATUS_W-1:0] status;
        logic                   invalid;
    } h9_score_packet_t;

    typedef struct packed {
        logic [H9_META_W-1:0]   token_meta;
        logic [H9_HEAD_W-1:0]   head_id;
        logic [H9_INDEX_W-1:0]  logical_token_index;
        logic [H9_INDEX_W-1:0]  cache_slot;
        logic [H9_INDEX_W-1:0]  probability_index;
        logic [31:0]            probability_fp32;
        logic                   last_probability;
        logic [H9_STATUS_W-1:0] status;
        logic                   invalid;
    } h9_probability_packet_t;
endpackage

`default_nettype wire
