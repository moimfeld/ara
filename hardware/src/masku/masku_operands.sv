// Copyright 2023 ETH Zurich and University of Bologna.
// Solderpad Hardware License, Version 0.51, see LICENSE for details.
// SPDX-License-Identifier: SHL-0.51
//
// Mask Unit Operands Module
//
// Author: Moritz Imfeld <moimfeld@student.ethz.ch>
//
//
// Description:
//  Module takes operands coming from the lanes and then unpacks and prepares them
//  for mask instruction execution.
//
//
// Incoming Operands:
// masku_operands_i = {v0.m, vs1, v2, alu_result, fpu_result}
//

module masku_operands import ara_pkg::*; import rvv_pkg::*; #(
    parameter int unsigned NrLanes = 0
  ) (
    // Control logic
    input masku_fu_e                        masku_fu_i,    // signal deciding from which functional unit the result should be taken from
    input pe_req_t                          vinsn_issue_i,
    input logic [idx_width(ELEN*NrLanes):0] vrf_pnt_i,

    // Operands coming from lanes
    input  elen_t [NrLanes-1:0][NrMaskFUnits+3-1:0] masku_operands_i,

    // Operands prepared for masku execution
    output elen_t [     NrLanes-1:0] masku_operand_alu_o,     // ALU/FPU result (shuffled, uncompressed)
    output logic  [NrLanes*ELEN-1:0] masku_operand_alu_seq_o, // ALU/FPU result (deshuffled, uncompressed)
    output elen_t [     NrLanes-1:0] masku_operand_vs1_o,     // vs1 (shuffled)
    output logic  [NrLanes*ELEN-1:0] masku_operand_vs1_seq_o, // vs1 (deshuffled)
    output elen_t [     NrLanes-1:0] masku_operand_vs2_o,     // vs2 (shuffled)
    output logic  [NrLanes*ELEN-1:0] masku_operand_vs2_seq_o, // vs2 (deshuffled)
    output elen_t [     NrLanes-1:0] masku_operand_m_o,       // Mask (shuffled)
    output logic  [NrLanes*ELEN-1:0] bit_enable_mask_o,       // Bit mask for mask unit instructions (shuffled like mask register)
    output logic  [NrLanes*ELEN-1:0] shuffled_vl_bit_mask_o,  // vl mask for mask unit instructions (first vl bits are 1, others 0)  (shuffled like mask register)
    output logic  [NrLanes*ELEN-1:0] alu_result_compressed_o  // ALU/FPU results compressed (from sew to 1-bit) (shuffled, in mask format)
  );

  // Imports
  import cf_math_pkg::idx_width;

  // Local Parameter
  localparam int unsigned DATAPATH_WIDTH = NrLanes * ELEN; // Mask Unit datapath width
  localparam int unsigned ELEN_BYTES     = ELEN / 8;

  // Helper signals
  logic [DATAPATH_WIDTH-1:0] deshuffled_vl_bit_mask; // this bit enable signal is only dependent on vl
  logic [DATAPATH_WIDTH-1:0] shuffled_vl_bit_mask;   // this bit enable signal is only dependent on vl
  vew_e                      bit_enable_shuffle_eew;

  // Extract operands from input (input comes in "shuffled form" from the lanes)
  for (genvar lane = 0; lane < NrLanes; lane++) begin
    assign masku_operand_m_o[lane]   = masku_operands_i[lane][0];
    assign masku_operand_vs1_o[lane] = masku_operands_i[lane][1];
    assign masku_operand_vs2_o[lane] = masku_operands_i[lane][2];
    assign masku_operand_alu_o[lane] = masku_operands_i[lane][3 + masku_fu_i];
  end

  // ---------------------
  // Deshuffle vs1 and vs2
  // ---------------------
  always_comb begin
      for (int b = 0; b < (NrLanes * ELEN_BYTES); b++) begin
        automatic int deshuffle_idx = deshuffle_index(b, NrLanes, vinsn_issue_i.vtype.vsew);
        automatic int lane_idx    = b / ELEN_BYTES; // rounded down to nearest integer
        automatic int lane_offset = b % ELEN_BYTES;
        masku_operand_alu_seq_o[8*deshuffle_idx +: 8] = masku_operand_vs1_o[lane_idx][8*lane_offset +: 8];
        masku_operand_vs1_seq_o[8*deshuffle_idx +: 8] = masku_operand_vs1_o[lane_idx][8*lane_offset +: 8];
        masku_operand_vs2_seq_o[8*deshuffle_idx +: 8] = masku_operand_vs2_o[lane_idx][8*lane_offset +: 8];
      end
  end

  // ------------------------------------------------
  // Generate shuffled and unshuffled bit level masks
  // ------------------------------------------------
  always_comb begin
    // Default assignments
    deshuffled_vl_bit_mask = '0;
    shuffled_vl_bit_mask   = '0;
    bit_enable_mask_o      = '0;

    // Generate bit enable depending on vector length
    if (vinsn_issue_i.vl < DATAPATH_WIDTH) begin
      deshuffled_vl_bit_mask = (1'b1 << vinsn_issue_i.vl) - 1'b1; // bit enable is one for vl number of bits and afterwards its 0
    end else begin
      deshuffled_vl_bit_mask = '1;
    end

    // Generate shuffled bit level mask
    // Note: bit-level mask is dependent on vl and mask (if vm == 0)
    assign bit_enable_shuffle_eew = vinsn_issue_i.op inside {[VMFEQ:VMSGTU], [VMSGT:VMSBC]} ? vinsn_issue_i.vtype.vsew : vinsn_issue_i.eew_vd_op;

    for (int unsigned b = 0; b < NrLanes * ELEN_BYTES; b++) begin
      // local helper signals
      logic [idx_width(DATAPATH_WIDTH)-1:0] src_operand_byte_shuffle_index;
      logic [idx_width(DATAPATH_WIDTH)-1:0] mask_operand_byte_shuffle_index;
      logic [       idx_width(NrLanes)-1:0] mask_operand_byte_shuffle_lane_index;
      logic [    idx_width(ELEN_BYTES)-1:0] mask_operand_byte_shuffle_lane_offset;

      // get shuffle idices
      // Note: two types of shuffle indices are needed because the source operand and the
      //       mask register might not have the same effective element width (eew)
      src_operand_byte_shuffle_index        = shuffle_index(b, NrLanes, bit_enable_shuffle_eew);
      mask_operand_byte_shuffle_index       = shuffle_index(b, NrLanes, vinsn_issue_i.eew_vmask);
      mask_operand_byte_shuffle_lane_index  = mask_operand_byte_shuffle_index[idx_width(ELEN_BYTES) +: idx_width(NrLanes)];
      mask_operand_byte_shuffle_lane_offset = mask_operand_byte_shuffle_index[idx_width(ELEN_BYTES)-1:0];

      // shuffle bit enable
      shuffled_vl_bit_mask[8*src_operand_byte_shuffle_index +: 8] = deshuffled_vl_bit_mask[8*b +: 8];

      // Generate bit-level mask
      bit_enable_mask_o[8*src_operand_byte_shuffle_index +: 8] = shuffled_vl_bit_mask[8*src_operand_byte_shuffle_index +: 8];
      if (!vinsn_issue_i.vm) begin
        bit_enable_mask_o[8*src_operand_byte_shuffle_index +: 8] &= masku_operand_m_o[mask_operand_byte_shuffle_lane_index][8*mask_operand_byte_shuffle_lane_offset +: 8];
      end
    end
  end

  assign shuffled_vl_bit_mask_o = shuffled_vl_bit_mask;

  // -------------------------------------------
  // Compress ALU/FPU results into a mask vector
  // -------------------------------------------
  always_comb begin
    alu_result_compressed_o = '0;
    for (int b = 0; b < (8 >> vinsn_issue_i.vtype.vsew) * NrLanes; b++) begin
      automatic int src_byte        = shuffle_index((1 << vinsn_issue_i.vtype.vsew) * b, NrLanes, vinsn_issue_i.vtype.vsew);
      automatic int src_byte_lane   = src_byte[idx_width(ELEN_BYTES) +: idx_width(NrLanes)];
      automatic int src_byte_offset = src_byte[idx_width(ELEN_BYTES)-1:0];

      automatic int dest_bit_seq  = b + vrf_pnt_i;
      automatic int dest_byte_seq = dest_bit_seq / ELEN_BYTES;
      automatic int dest_byte     = shuffle_index(dest_byte_seq, NrLanes, vinsn_issue_i.vtype.vsew);

      alu_result_compressed_o[ELEN_BYTES * dest_byte + dest_bit_seq[idx_width(ELEN_BYTES)-1:0]] = masku_operand_alu_o[src_byte_lane][8 * src_byte_offset];
    end
  end

endmodule : masku_operands
