// Copyright 2021 ETH Zurich and University of Bologna.
// Solderpad Hardware License, Version 0.51, see LICENSE for details.
// SPDX-License-Identifier: SHL-0.51
//
// Author: Matheus Cavalcante <matheusd@iis.ee.ethz.ch>
// Description:
// This is Ara's mask unit. It fetches operands from any one the lanes, and
// then sends back to them whether the elements are predicated or not.
// This unit is shared between all the functional units who can execute
// predicated instructions.

module masku import ara_pkg::*; import rvv_pkg::*; #(
    parameter  int  unsigned NrLanes = 0,
    parameter  type          vaddr_t = logic, // Type used to address vector register file elements
    // Dependant parameters. DO NOT CHANGE!
    localparam int  unsigned DataWidth = $bits(elen_t), // Width of the lane datapath
    localparam int  unsigned StrbWidth = DataWidth/8,
    localparam type          strb_t    = logic [StrbWidth-1:0] // Byte-strobe type
  ) (
    input  logic                                       clk_i,
    input  logic                                       rst_ni,
    // Interface with the main sequencer
    input  pe_req_t                                    pe_req_i,
    input  logic                                       pe_req_valid_i,
    input  logic     [NrVInsn-1:0]                     pe_vinsn_running_i,
    output logic                                       pe_req_ready_o,
    output pe_resp_t                                   pe_resp_o,
    output elen_t                                      result_scalar_o,
    output logic                                       result_scalar_valid_o,
    // Interface with the lanes
    input  elen_t    [NrLanes-1:0][NrMaskFUnits+3-1:0] masku_operand_i,
    input  logic     [NrLanes-1:0][NrMaskFUnits+3-1:0] masku_operand_valid_i,
    output logic     [NrLanes-1:0][NrMaskFUnits+3-1:0] masku_operand_ready_o,
    output logic     [NrLanes-1:0]                     masku_operand_req_o,
    output logic     [NrLanes-1:0]                     masku_result_req_o,
    output vid_t     [NrLanes-1:0]                     masku_result_id_o,
    output vaddr_t   [NrLanes-1:0]                     masku_result_addr_o,
    output elen_t    [NrLanes-1:0]                     masku_result_wdata_o,
    output strb_t    [NrLanes-1:0]                     masku_result_be_o,
    input  logic     [NrLanes-1:0]                     masku_result_gnt_i,
    input  logic     [NrLanes-1:0]                     masku_result_final_gnt_i,
    // Interface with the VFUs
    output strb_t    [NrLanes-1:0]                     mask_o,
    output logic     [NrLanes-1:0]                     mask_valid_o,
    output logic                                       mask_valid_lane_o,
    input  logic     [NrLanes-1:0]                     lane_mask_ready_i,
    input  logic                                       vldu_mask_ready_i,
    input  logic                                       vstu_mask_ready_i,
    input  logic                                       sldu_mask_ready_i
  );

  import cf_math_pkg::idx_width;

  // Pointers
  //
  // We need a pointer to which bit on the full VRF word we are reading mask operands from.
  logic [idx_width(DataWidth*NrLanes):0] mask_pnt_d, mask_pnt_q;
  // We need a pointer to which bit on the full VRF word we are writing results to.
  logic [idx_width(DataWidth*NrLanes):0] vrf_pnt_d, vrf_pnt_q;

  // Remaining elements of the current instruction in the read operand phase
  vlen_t read_cnt_d, read_cnt_q;
  // Remaining elements of the current instruction in the issue phase
  vlen_t issue_cnt_d, issue_cnt_q;
  // Remaining elements of the current instruction in the commit phase
  vlen_t commit_cnt_d, commit_cnt_q;

  ////////////////
  //  Operands  //
  ////////////////

  // Information about which is the target FU of the request
  masku_fu_e masku_operand_fu;

  // ALU/FPU result (shuffled)
  elen_t [NrLanes-1:0] masku_operand_a;
  logic  [NrLanes-1:0] masku_operand_a_valid_i;
  logic  [NrLanes-1:0] masku_operand_a_ready_o;

  // ALU/FPU result (deshuffled)
  logic  [NrLanes*ELEN-1:0] masku_operand_a_seq;

  // vs1 (shuffeld)
  elen_t [NrLanes-1:0] masku_operand_vs1;
  logic  [NrLanes-1:0] masku_operand_vs1_valid_i;
  logic  [NrLanes-1:0] masku_operand_vs1_ready_o;

  // vs1 (deshuffled)
  logic  [NrLanes*ELEN-1:0] masku_operand_vs1_seq;

  // vs2 (shuffled)
  elen_t [NrLanes-1:0] masku_operand_vs2;
  logic  [NrLanes-1:0] masku_operand_vs2_valid_i;
  logic  [NrLanes-1:0] masku_operand_vs2_ready_o;

  // vs2 (deshuffled)
  logic  [NrLanes*ELEN-1:0] masku_operand_vs2_seq;
  
  // Mask
  elen_t [NrLanes-1:0] masku_operand_m;
  logic  [NrLanes-1:0] masku_operand_m_valid_i;
  logic  [NrLanes-1:0] masku_operand_m_ready_o;

  // Mask deshuffled
  logic  [NrLanes*ELEN-1:0] masku_operand_m_seq;

  // Insn-queue related signal
  pe_req_t vinsn_issue;

  logic  [NrLanes*ELEN-1:0] bit_enable_mask;
  logic  [NrLanes*ELEN-1:0] bit_enable_shuffle;
  logic  [NrLanes*ELEN-1:0] alu_result_compressed;

  // Performs all shuffling and deshuffling of mask operands (including masks for mask instructions) - purely combinational module
  masku_operands #(
    .NrLanes ( NrLanes )
  ) i_masku_operands (
    // Control logic
    .masku_fu_i              (      masku_operand_fu ),
    .vinsn_issue_i           (           vinsn_issue ),
    .vrf_pnt_i               (             vrf_pnt_q ),
    // Operands coming from lanes
    .masku_operands_i        (       masku_operand_i ),
    // Operands prepared for mask unit execution
    .masku_operand_alu_o     (       masku_operand_a ),
    .masku_operand_alu_seq_o (   masku_operand_a_seq ),
    .masku_operand_vs1_o     (     masku_operand_vs1 ),
    .masku_operand_vs1_seq_o ( masku_operand_vs1_seq ),
    .masku_operand_vs2_o     (     masku_operand_vs2 ),
    .masku_operand_vs2_seq_o ( masku_operand_vs2_seq ),
    .masku_operand_m_o       (       masku_operand_m ),
    .masku_operand_m_seq_o   (   masku_operand_m_seq ),
    .bit_enable_mask_o       (       bit_enable_mask ),
    .shuffled_vl_bit_mask_o  (    bit_enable_shuffle ),
    .alu_result_compressed_o ( alu_result_compressed )
  );


  // Local Parameter W_CPOP and W_VFIRST
  //
  // Description: Parameters W_CPOP and W_VFIRST enable time multiplexing of vcpop.m and vfirst.m instruction.
  //
  // Legal range W_CPOP:   {64, 128, ... , DataWidth*NrLanes} // DataWidth = 64
  // Legal range W_VFIRST: {64, 128, ... , DataWidth*NrLanes} // DataWidth = 64
  //
  // Execution time example for vcpop.m (similar for vfirst.m):
  // W_CPOP = 64; VLEN = 1024; vl = 1024
  // t_vcpop.m = VLEN/W_CPOP = 8 [Cycles]
  localparam int W_CPOP   = 64;
  localparam int W_VFIRST = 64;
  // derived parameters
  localparam int MAX_W_CPOP_VFIRST = (W_CPOP > W_VFIRST) ? W_CPOP : W_VFIRST;
  localparam int N_SLICES_CPOP   = NrLanes * DataWidth / W_CPOP;
  localparam int N_SLICES_VFIRST = NrLanes * DataWidth / W_VFIRST;
  // Check if parameters are within range
  if (((W_CPOP & (W_CPOP - 1)) != 0) || (W_CPOP < 64)) begin
    $fatal(1, "Parameter W_CPOP must be power of 2.");
  end else if (((W_VFIRST & (W_VFIRST - 1)) != 0) || (W_VFIRST < 64)) begin
    $fatal(1, "Parameter W_VFIRST must be power of 2.");
  end

  // VFIRST and VCPOP Signals
  logic  [NrLanes*ELEN-1:0]              vcpop_operand;
  logic  [$clog2(W_VFIRST):0]            popcount;
  logic  [$clog2(VLEN):0]                popcount_d, popcount_q;
  logic  [$clog2(W_VFIRST)-1:0]          vfirst_count;
  logic  [$clog2(VLEN)-1:0]              vfirst_count_d, vfirst_count_q;
  logic                                  vfirst_empty;
  // counter to keep track of how many slices of the vcpop_operand have been processed
  logic [$clog2(MAX_W_CPOP_VFIRST):0]   vcpop_slice_cnt_d, vcpop_slice_cnt_q;
  logic [W_CPOP-1:0]                    vcpop_slice;
  logic [W_VFIRST-1:0]                  vfirst_slice;


  // VCOMPRESS and VRGATHER signals
  logic [            NrLanes*ELEN-1:0] vcomp_vrgath_result_d, vcomp_vrgath_result_q, vcomp_vrgath_result_shuffled;
  logic [idx_width(NrLanes*ELENB)-1:0] vcomp_vrgath_result_vec_elements; // counts number of elements in result vector
  logic [  idx_width(NrLanes*ELENB):0] total_input_elements; // Helper control signal to clean code (number of elements per masku_operand - depends on vsew and NrLanes)
  logic [          idx_width(MAXVL):0] current_vlmax; // Helper singnal that is set to vlmax for the current SEW setting
  logic [          idx_width(MAXVL):0] vcomp_vrgath_result_element_cnt_d, vcomp_vrgath_result_element_cnt_q; // counts total number of result elements
  logic [          idx_width(MAXVL):0] vcomp_vrgath_processed_element_vs1_cnt_d, vcomp_vrgath_processed_element_vs1_cnt_q; // counts number of elements processed by vrgather instructions - unused by vcompress
  logic [          idx_width(MAXVL):0] vcomp_vrgath_processed_element_vs2_cnt_d, vcomp_vrgath_processed_element_vs2_cnt_q; // counts number of elements processed by vcompress/vrgather instructions
  logic                                vcomp_vrgath_result_valid; // result buffer can be written back to vreg
  logic                                vcompress_stall_vs2_ready; // stall fetching mechanism for vs2
  logic                                vcompress_finished, vrgather_finished; // vcompress or vrgather instruction finished
  logic [           NrLanes*ELENB-1:0] vrgath_be; // byte enable for vrgather

  // Control flow for mask operands
  for (genvar lane = 0; lane < NrLanes; lane++) begin: gen_unpack_masku_operands
    // immediately acknowledge operands coming from functional units
    assign masku_operand_a_valid_i[lane] = masku_operand_valid_i[lane][3 + masku_operand_fu];
    for (genvar operand_fu = 0; operand_fu < NrMaskFUnits; operand_fu++) begin: gen_masku_operand_ready
      assign masku_operand_ready_o[lane][3 + operand_fu] = (masku_fu_e'(operand_fu) == masku_operand_fu) && masku_operand_a_ready_o[lane];
    end: gen_masku_operand_ready

    // acknowledge vs2 if there is no stall.
    assign masku_operand_vs2_valid_i[lane] = masku_operand_valid_i[lane][2];
    assign masku_operand_ready_o[lane][2]  = masku_operand_vs2_ready_o[lane];

    assign masku_operand_vs1_valid_i[lane] = (vinsn_issue.op inside {[VMSBF:VID]}) ? '1 : masku_operand_valid_i[lane][1];
    assign masku_operand_ready_o[lane][1]  = masku_operand_vs1_ready_o[lane];

    assign masku_operand_m_valid_i[lane]   = masku_operand_valid_i[lane][0];
    assign masku_operand_ready_o[lane][0]  = masku_operand_m_ready_o[lane];
  end: gen_unpack_masku_operands


  ////////////////////////////////
  //  Vector instruction queue  //
  ////////////////////////////////

  // We store a certain number of in-flight vector instructions.
  // To avoid any hazards between masked vector instructions, the mask
  // unit is only capable of handling one vector instruction at a time.
  // Optimizing this unit is left as future work.

  localparam VInsnQueueDepth = MaskuInsnQueueDepth;

  struct packed {
    pe_req_t [VInsnQueueDepth-1:0] vinsn;

    // We also need to count how many instructions are queueing to be
    // issued/committed, to avoid accepting more instructions than
    // we can handle.
    logic [idx_width(VInsnQueueDepth)-1:0] issue_cnt;
    logic [idx_width(VInsnQueueDepth)-1:0] commit_cnt;
  } vinsn_queue_d, vinsn_queue_q;

  // Is the vector instruction queue full?
  logic vinsn_queue_full;
  assign vinsn_queue_full = (vinsn_queue_q.commit_cnt == VInsnQueueDepth);

  // Do we have a vector instruction ready to be issued?
  logic    vinsn_issue_valid;
  assign vinsn_issue       = vinsn_queue_q.vinsn[0];
  assign vinsn_issue_valid = (vinsn_queue_q.issue_cnt != '0);

  // Do we have a vector instruction with results being committed?
  pe_req_t vinsn_commit;
  logic    vinsn_commit_valid;
  assign vinsn_commit       = vinsn_queue_q.vinsn[0];
  assign vinsn_commit_valid = (vinsn_queue_q.commit_cnt != '0);

  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      vinsn_queue_q <= '0;
    end else begin
      vinsn_queue_q <= vinsn_queue_d;
    end
  end

  ///////////////////
  //  Mask queues  //
  ///////////////////

  localparam int unsigned MaskQueueDepth = 2;

  // There is a mask queue per lane, holding the operands that were not
  // yet used by the corresponding lane.

  // Mask queue
  strb_t [MaskQueueDepth-1:0][NrLanes-1:0] mask_queue_d, mask_queue_q;
  logic  [MaskQueueDepth-1:0][NrLanes-1:0] mask_queue_valid_d, mask_queue_valid_q;
  // We need two pointers in the mask queue. One pointer to
  // indicate with `strb_t` we are currently writing into (write_pnt),
  // and one pointer to indicate which `strb_t` we are currently
  // reading from and writing into the lanes (read_pnt).
  logic  [idx_width(MaskQueueDepth)-1:0]   mask_queue_write_pnt_d, mask_queue_write_pnt_q;
  logic  [idx_width(MaskQueueDepth)-1:0]   mask_queue_read_pnt_d, mask_queue_read_pnt_q;
  // We need to count how many valid elements are there in this mask queue.
  logic  [idx_width(MaskQueueDepth):0]     mask_queue_cnt_d, mask_queue_cnt_q;

  // Is the mask queue full?
  logic mask_queue_full;
  assign mask_queue_full = (mask_queue_cnt_q == MaskQueueDepth);
  // Is the mask queue empty?
  logic mask_queue_empty;
  assign mask_queue_empty = (mask_queue_cnt_q == '0);

  always_ff @(posedge clk_i or negedge rst_ni) begin: p_mask_queue_ff
    if (!rst_ni) begin
      mask_queue_q           <= '0;
      mask_queue_valid_q     <= '0;
      mask_queue_write_pnt_q <= '0;
      mask_queue_read_pnt_q  <= '0;
      mask_queue_cnt_q       <= '0;
    end else begin
      mask_queue_q           <= mask_queue_d;
      mask_queue_valid_q     <= mask_queue_valid_d;
      mask_queue_write_pnt_q <= mask_queue_write_pnt_d;
      mask_queue_read_pnt_q  <= mask_queue_read_pnt_d;
      mask_queue_cnt_q       <= mask_queue_cnt_d;
    end
  end

  /////////////////////
  //  Result queues  //
  /////////////////////

  localparam int unsigned ResultQueueDepth = 2;

  // There is a result queue per lane, holding the results that were not
  // yet accepted by the corresponding lane.
  typedef struct packed {
    vid_t id;
    vaddr_t addr;
    elen_t wdata;
    strb_t be;
  } payload_t;

  // Result queue
  payload_t [ResultQueueDepth-1:0][NrLanes-1:0] result_queue_d, result_queue_q;
  logic     [ResultQueueDepth-1:0][NrLanes-1:0] result_queue_valid_d, result_queue_valid_q;
  // We need two pointers in the result queue. One pointer to
  // indicate with `payload_t` we are currently writing into (write_pnt),
  // and one pointer to indicate which `payload_t` we are currently
  // reading from and writing into the lanes (read_pnt).
  logic     [idx_width(ResultQueueDepth)-1:0]   result_queue_write_pnt_d, result_queue_write_pnt_q;
  logic     [idx_width(ResultQueueDepth)-1:0]   result_queue_read_pnt_d, result_queue_read_pnt_q, result_queue_read_pnt_m;
  // We need to count how many valid elements are there in this result queue.
  logic     [idx_width(ResultQueueDepth):0]     result_queue_cnt_d, result_queue_cnt_q;
  // Vector to register the final grants from the operand requesters, which indicate
  // that the result was actually written in the VRF (while the normal grant just says
  // that the result was accepted by the operand requester stage
  logic     [NrLanes-1:0]                       result_final_gnt_d, result_final_gnt_q;

  // Is the result queue full?
  logic result_queue_full;
  assign result_queue_full = (result_queue_cnt_q == ResultQueueDepth);
  // Is the result queue empty?
  logic result_queue_empty;
  assign result_queue_empty = (result_queue_cnt_q == '0);

  // vmsbf, vmsif, vmsof, viota, vid, vcpop, vfirst variables
  logic  [NrLanes*DataWidth-1:0] alu_result_f, alu_result_ff;
  logic  [NrLanes*DataWidth-1:0] alu_operand_a, alu_operand_a_seq, alu_operand_a_seq_f;
  logic  [NrLanes*DataWidth-1:0] alu_operand_b_seq, alu_operand_b_seq_m, alu_operand_b_seq_f, alu_operand_b_seq_ff;
  logic  [NrLanes*DataWidth-1:0] alu_result_vm, alu_result_vm_m, alu_result_vm_seq;
  logic  [NrLanes*DataWidth-1:0] alu_src_idx, alu_src_idx_m;
  logic  [4:0]                   iteration_count_d, iteration_count_q;
  logic                          not_found_one_d, not_found_one_q;

  always_ff @(posedge clk_i or negedge rst_ni) begin: p_result_queue_ff
    if (!rst_ni) begin
      result_queue_q           <= '0;
      result_queue_valid_q     <= '0;
      result_queue_write_pnt_q <= '0;
      result_queue_read_pnt_q  <= '0;
      result_queue_read_pnt_m  <= '0;
      result_queue_cnt_q       <= '0;
      alu_result_f             <= '0;
      alu_result_ff            <= '0;
      not_found_one_q          <= 1'b1;
      alu_operand_b_seq_f      <= '0;
      alu_operand_b_seq_ff     <= '0;
      iteration_count_q        <= '0;
    end else begin
      result_queue_q           <= result_queue_d;
      result_queue_valid_q     <= result_queue_valid_d;
      result_queue_write_pnt_q <= result_queue_write_pnt_d;
      result_queue_read_pnt_m  <= result_queue_write_pnt_q;
      result_queue_read_pnt_q  <= (vinsn_issue.op inside {[VMSBF:VID]}) ? result_queue_read_pnt_m : result_queue_read_pnt_d;
      result_queue_cnt_q       <= result_queue_cnt_d;
      alu_result_f             <= (pe_req_ready_o) ? '0 : (!vinsn_issue.vm) ? alu_result_vm : alu_result_vm_seq;
      alu_result_ff            <= alu_result_f;
      not_found_one_q          <= not_found_one_d;
      alu_operand_b_seq_f      <= (pe_req_ready_o) ? '0 : alu_operand_b_seq_m;
      alu_operand_b_seq_ff     <= alu_operand_b_seq_f;
      iteration_count_q        <= iteration_count_d;
    end
  end

  // iteration count for masked instrctions
  always_comb begin
    if (vinsn_issue_valid && &masku_operand_a_valid_i) begin
      iteration_count_d = iteration_count_q + 1'b1;
    end else begin
      iteration_count_d = iteration_count_q;
    end
    if (pe_req_ready_o && !vinsn_issue_valid) begin
      iteration_count_d = '0;
    end
  end

  ////////////////////////////
  //// Scalar result reg  ////
  ////////////////////////////

  elen_t result_scalar_d;
  logic  result_scalar_valid_d;

  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      result_scalar_o       <= '0;
      result_scalar_valid_o <= '0;
    end else begin
      result_scalar_o       <= result_scalar_d;
      result_scalar_valid_o <= result_scalar_valid_d;
    end
  end

  ////////////////
  //  Mask ALU  //
  ////////////////

  elen_t [NrLanes-1:0]                   alu_result;
  logic  [NrLanes*ELEN-1:0]              mask;

  // keep track if first 1 mask element was found
  logic vfirst_found;

  // assign operand slices to be processed by popcount and lzc
  assign vcpop_slice  = vcpop_operand[(vcpop_slice_cnt_q * W_CPOP) +: W_CPOP];
  assign vfirst_slice = vcpop_operand[(vcpop_slice_cnt_q * W_VFIRST) +: W_VFIRST];

  // Population count for vcpop.m instruction
  popcount #(
    .INPUT_WIDTH (W_CPOP)
  ) i_popcount (
    .data_i    (vcpop_slice),
    .popcount_o(popcount     )
  );

  // Trailing zero counter
  lzc #(
    .WIDTH(W_VFIRST),
    .MODE (0)
  ) i_clz (
    .in_i    (vfirst_slice ),
    .cnt_o   (vfirst_count ),
    .empty_o (vfirst_empty )
  );

  always_comb begin: p_mask_alu
    alu_result          = '0;
    not_found_one_d     = pe_req_ready_o ? 1'b1 : not_found_one_q;
    alu_result_vm       = '0;
    alu_result_vm_m     = '0;
    alu_result_vm_seq   = '0;
    alu_operand_b_seq   = '0;
    alu_operand_b_seq_m = '0;
    mask                = '0;
    vcpop_operand       = '0;
    vcompress_finished                       = 1'b0;
    vrgather_finished                        = 1'b0;
    total_input_elements                     = '0;
    current_vlmax                            = '0;
    vcomp_vrgath_result_vec_elements         = '0;
    vcomp_vrgath_result_element_cnt_d        = '0;
    vcomp_vrgath_processed_element_vs1_cnt_d = '0;
    vcomp_vrgath_processed_element_vs2_cnt_d = '0;
    vcomp_vrgath_result_d                    = '0;
    vcomp_vrgath_result_shuffled             = '0;
    vcomp_vrgath_result_valid                = '0;
    vcompress_stall_vs2_ready                = '0;
    vrgath_be                                = '0;
    masku_operand_req_o                      = '0;
    masku_operand_vs1_ready_o                = '0;
    masku_operand_vs2_ready_o                = '0;

    if (vinsn_issue_valid) begin

      alu_operand_a     = masku_operand_a;
      alu_operand_b_seq = masku_operand_a_seq;

      // Mask generation
      unique case (vinsn_issue.op) inside
        [VMSBF:VID] :
          if (&masku_operand_a_valid_i) begin
            unique case (vinsn_issue.vtype.vsew)
              EW8 : for (int i = 0; i < (DataWidth * NrLanes)/8; i++)
                      mask [(i*8) +: 8]   = {8{bit_enable_mask [i+(((DataWidth * NrLanes)/8)*(iteration_count_d-1))]}};
              EW16: for (int i = 0; i < (DataWidth * NrLanes)/16; i++)
                      mask [(i*16) +: 16] = {16{bit_enable_mask [i+(((DataWidth * NrLanes)/16)*(iteration_count_d-1))]}};
              EW32: for (int i = 0; i < (DataWidth * NrLanes)/32; i++)
                      mask [(i*32) +: 32] = {32{bit_enable_mask [i+(((DataWidth * NrLanes)/32)*(iteration_count_d-1))]}};
              EW64: for (int i = 0; i < (DataWidth * NrLanes)/64; i++)
                      mask [(i*64) +: 64] = {64{bit_enable_mask [i+(((DataWidth * NrLanes)/64)*(iteration_count_d-1))]}};
            endcase
          end else begin
            mask = '0;
          end
        default:;
      endcase

      // Evaluate the instruction
      unique case (vinsn_issue.op) inside
        [VMANDNOT:VMXNOR]: alu_result = (masku_operand_a) | (~bit_enable_shuffle);
        [VMFEQ:VMSGTU], [VMSGT:VMSBC]:  alu_result = (alu_result_compressed & bit_enable_mask) | (~bit_enable_shuffle);
        [VMSBF:VMSIF] : begin
            if (&masku_operand_a_valid_i) begin
                for (int i = 0; i < NrLanes * DataWidth; i++) begin
                    if (alu_operand_b_seq[i] == 1'b0) begin
                        alu_result_vm[i] = (vinsn_issue.op == VMSOF) ? 1'b0 : not_found_one_d;
                    end else begin
                        not_found_one_d = 1'b0;
                        alu_result_vm[i] = (vinsn_issue.op == VMSBF) ? not_found_one_d : 1'b1;
                        break;
                    end
                end
                alu_result_vm_m = (!vinsn_issue.vm) ? alu_result_vm & bit_enable_mask : alu_result_vm;
            end else begin
                alu_result_vm = '0;
            end
        end
        VIOTA: begin
          if (&masku_operand_a_valid_i) begin
            alu_operand_b_seq_m = alu_operand_b_seq & bit_enable_mask;
            unique case (vinsn_issue.vtype.vsew)
              EW8 : begin
                if (issue_cnt_q < vinsn_issue.vl) begin
                  alu_result_vm [7:0] = alu_operand_b_seq_ff [(NrLanes*DataWidth)-1-:8] + alu_result_ff [(NrLanes*DataWidth)-1-:8];
                end else begin
                  alu_result_vm [7:0] = '0;
                end
                for (int index = 1; index < (NrLanes*DataWidth)/8; index++) begin
                  alu_result_vm   [(index*8) +: 7] = alu_operand_b_seq_m [index-1] + alu_result_vm [((index-1)*8) +: 7];
                  alu_result_vm_m [(index*8) +: 7] = alu_result_vm [(index*8) +: 7];
                end
              end
              EW16: begin
                if (issue_cnt_q < vinsn_issue.vl) begin
                  alu_result_vm [15:0] = alu_operand_b_seq_ff [(NrLanes*DataWidth)-1-:16] + alu_result_ff [(NrLanes*DataWidth)-1-:16];
                end else begin
                  alu_result_vm [15:0] = '0;
                end
                for (int index = 1; index < (NrLanes*DataWidth)/16; index++) begin
                  alu_result_vm   [(index*16) +: 15] = alu_operand_b_seq_m [index-1] + alu_result_vm [((index-1)*16) +: 15];
                  alu_result_vm_m [(index*16) +: 15] = alu_result_vm [(index*16) +: 15];
                end
              end
              EW32: begin
                if (issue_cnt_q < vinsn_issue.vl) begin
                  alu_result_vm [31:0] = alu_operand_b_seq_ff [(NrLanes*DataWidth)-1-:32] + alu_result_ff [(NrLanes*DataWidth)-1-:32];
                end else begin
                  alu_result_vm [31:0] = '0;
                end
                for (int index = 1; index < (NrLanes*DataWidth)/32; index++) begin
                  alu_result_vm   [(index*32) +: 31] = alu_operand_b_seq_m [index-1] + alu_result_vm [((index-1)*32) +: 31];
                  alu_result_vm_m [(index*32) +: 31] = alu_result_vm [(index*32) +: 31];
                end
              end
              EW64: begin
                if (issue_cnt_q < vinsn_issue.vl) begin
                  alu_result_vm [63:0] = alu_operand_b_seq_ff [(NrLanes*DataWidth)-1-:64] + alu_result_ff [(NrLanes*DataWidth)-1-:64];
                end else begin
                  alu_result_vm [63:0] = '0;
                end
                for (int index = 1; index < (NrLanes*DataWidth)/64; index++) begin
                  alu_result_vm   [(index*64) +: 63] = alu_operand_b_seq_m [index-1] + alu_result_vm [((index-1)*64) +: 63];
                  alu_result_vm_m [(index*64) +: 63] = alu_result_vm [(index*64) +: 63];
                end
              end
            endcase
          end
        end
        VID: begin
          if (&masku_operand_a_valid_i) begin
            unique case (vinsn_issue.vtype.vsew)
              EW8 : begin
                for (int index = 1; index < (NrLanes*DataWidth)/8; index++) begin
                  alu_result_vm [(index*8) +: 7] = (((NrLanes * DataWidth)/8) >= vinsn_issue.vl) ? index : index-(((vinsn_issue.vl/((NrLanes * DataWidth)/8))-iteration_count_d)*32);
                  alu_result_vm_m = alu_result_vm & mask;
                end
              end
              EW16: begin
                for (int index = 1; index < (NrLanes*DataWidth)/16; index++) begin
                  alu_result_vm [(index*16) +: 15] = (((NrLanes * DataWidth)/8) >= vinsn_issue.vl) ? index : index-(((vinsn_issue.vl/((NrLanes * DataWidth)/8))-iteration_count_d)*16);
                  alu_result_vm_m = alu_result_vm & mask;
                end
              end
              EW32: begin
                for (int index = 1; index < (NrLanes*DataWidth)/32; index++) begin
                  alu_result_vm [(index*32) +: 31] = (((NrLanes * DataWidth)/8) >= vinsn_issue.vl) ? index : index-(((vinsn_issue.vl/((NrLanes * DataWidth)/8))-iteration_count_d)*8);
                  alu_result_vm_m = alu_result_vm & mask;
                end
              end
              EW64: begin
                for (int index = 1; index < (NrLanes*DataWidth)/64; index++) begin
                  alu_result_vm [(index*64) +: 63] = (((NrLanes * DataWidth)/8) >= vinsn_issue.vl) ? index : index-(((vinsn_issue.vl/((NrLanes * DataWidth)/8))-iteration_count_d)*4);
                  alu_result_vm_m = alu_result_vm & mask;
                end
              end
            endcase
          end
        end
        [VCPOP:VFIRST] : begin
          vcpop_operand = (!vinsn_issue.vm) ? masku_operand_a & bit_enable_mask : masku_operand_a;
        end
        // NOTE: Current VCOMPRESS implementation assumes that result queue is always ready to accept results! If result queue is not ready for the result,
        //       this implementation will break
        VCOMPRESS : begin
          // Preserve vcompress state (because one result vector "slice" can take multiple cycles to compute)
          vcomp_vrgath_result_element_cnt_d        = vcomp_vrgath_result_element_cnt_q;
          vcomp_vrgath_processed_element_vs2_cnt_d = vcomp_vrgath_processed_element_vs2_cnt_q;
          vcomp_vrgath_result_d                    = vcomp_vrgath_result_q;

          // initialize control signals
          vcomp_vrgath_result_valid = 1'b0;
          vcompress_stall_vs2_ready = 1'b0;
          masku_operand_vs1_ready_o = '0;
          masku_operand_vs2_ready_o = '0;
          total_input_elements      = (NrLanes * ELEN) / (8 << vinsn_issue.vtype.vsew);

          // Assign vcomp_vrgath_result_vec_elements (# valid result elements in result vector)
          vcomp_vrgath_result_vec_elements = vcomp_vrgath_result_element_cnt_d % total_input_elements;

          // After result writeback, reset vcomp_vrgath_result_d
          if (vcomp_vrgath_result_vec_elements == '0) begin
            vcomp_vrgath_result_d = '0;
          end

          // Only start/continue computing vcompress result if both operands are ready
          if ((&masku_operand_vs1_valid_i) && (&masku_operand_vs2_valid_i)) begin
            masku_operand_vs2_ready_o = '1; // by default, receive new operand (only if vcomp_vrgath_result_d runs full keep )
            // Iterate over every element of the incoming vs2 vector section and copy active elements to destination vector
            for (int i = 0; i < total_input_elements; i++) begin
              if (masku_operand_vs1_seq[vcomp_vrgath_processed_element_vs2_cnt_d+i]) begin
                // copy element from source vector to (compressed) destination vector (ONLY IF vcomp_vrgath_result_d IS NOT FULL)
                if (!vcomp_vrgath_result_valid) begin
                  unique case (vinsn_issue.vtype.vsew)
                    EW8 : vcomp_vrgath_result_d[vcomp_vrgath_result_vec_elements* 8 +:  8] = masku_operand_vs2_seq[i* 8 +:  8];
                    EW16: vcomp_vrgath_result_d[vcomp_vrgath_result_vec_elements*16 +: 16] = masku_operand_vs2_seq[i*16 +: 16];
                    EW32: vcomp_vrgath_result_d[vcomp_vrgath_result_vec_elements*32 +: 32] = masku_operand_vs2_seq[i*32 +: 32];
                    EW64: vcomp_vrgath_result_d[vcomp_vrgath_result_vec_elements*64 +: 64] = masku_operand_vs2_seq[i*64 +: 64];
                    default: ; // Not sure what should be the default
                  endcase
                  vcomp_vrgath_result_element_cnt_d += 1'b1; // Note: Assignment depends on previous iteration of the for-loop --> this loop generates long combinational paths (might be suboptimal for timing)
                  vcomp_vrgath_result_vec_elements  += 1'b1; // update vcomp_vrgath_result_vec_elements
                end else begin
                  vcompress_stall_vs2_ready = 1'b1; // if there is another active element after the vcomp_vrgath_result_d is full, we must keep vs2 for another cycle, such that these active elements are not lost
                end

                // Check if vcomp_vrgath_result_d is full --> if full, result should be written back to registerfile
                vcomp_vrgath_result_valid = vcomp_vrgath_result_vec_elements == total_input_elements;
              end
            end
            // MOIMFELD: This line can possibly still create issues, because it gets incremented even if not all elements have been processed
            vcomp_vrgath_processed_element_vs2_cnt_d += total_input_elements;

            // Update control signals
            if (vcompress_stall_vs2_ready) begin
              masku_operand_vs2_ready_o = '0; // do not request new operand if old operand was not fully processed
            end

            // For vcompress, the vs2 counter also reflects how many elements were processed of vs1 (unlike for vrgather where an extra counter is needed)
            if ((vcomp_vrgath_processed_element_vs2_cnt_d % (NrLanes * ELEN)) == '0) begin
              masku_operand_vs1_ready_o = '1;
            end

            // Write back final elements REMOVE MAYBE
            // if (vcomp_vrgath_processed_element_vs2_cnt_d >= vinsn_issue.vl && vcomp_vrgath_result_vec_elements >= '0) begin
            //   vcomp_vrgath_result_valid = 1'b1; // write back last result if result vector is not empty
            // end

            // Check if vcompressed instrcution is finished
            if (vcomp_vrgath_processed_element_vs2_cnt_d >= vinsn_issue.vl) begin
              vcompress_finished = 1'b1;
              masku_operand_vs1_ready_o = '1; // acknowledge last operand
              masku_operand_vs2_ready_o = '1; // acknowledge last operand
              if (vcomp_vrgath_result_vec_elements >= '0) begin
                vcomp_vrgath_result_valid = 1'b1; // write back last result if result vector is not empty
              end
            end
          end
        end
        VRGATHER: begin
          // Preserve vgather state (because one result vector "slice" can take multiple cycles to compute)
          vcomp_vrgath_result_element_cnt_d        = vcomp_vrgath_result_element_cnt_q;
          vcomp_vrgath_processed_element_vs1_cnt_d = vcomp_vrgath_processed_element_vs1_cnt_q;
          vcomp_vrgath_processed_element_vs2_cnt_d = vcomp_vrgath_processed_element_vs2_cnt_q;
          vcomp_vrgath_result_d                    = vcomp_vrgath_result_q;

          // initialize control signals
          vcomp_vrgath_result_valid = 1'b0;
          masku_operand_vs1_ready_o = '0;
          masku_operand_vs2_ready_o = '0;
          total_input_elements      = (NrLanes * ELEN) / (8 << vinsn_issue.vtype.vsew); // number of elements in input slice
          vrgath_be                 = '0;
          if (vinsn_issue.vtype.vlmul[2]) begin // this if statement checks if LMUL is a fraction (i.e. 1/2, 1/4 or 1/8)
            current_vlmax = (MAXVL >> (11 - vinsn_issue.vtype.vlmul[2:0])) >> vinsn_issue.vtype.vsew; // compute current_vlmax for LMUL = 1/2 or 1/4 or 1/8
          end else begin
            current_vlmax = (MAXVL >> (3  - vinsn_issue.vtype.vlmul[1:0])) >> vinsn_issue.vtype.vsew; // compute current_vlmax for LMUL = 1, 2, 4, 8
          end
          // Only start/continue computing vcompress result once both operands are ready
          if ((&masku_operand_vs1_valid_i) && (&masku_operand_vs2_valid_i)) begin
            
            // Update control signals
            masku_operand_vs2_ready_o = '1; // By default, acknowledge masku_operand_vs2 (if it is valid)

            // Perform gather operation by checking if any element index given by v1 exists in current vs2 slice
            for (int i = 0; i < total_input_elements; i++) begin
              // Extract indeces from vs1 vector slice
              automatic logic [ELEN-1:0] elem_idx = '0;
              unique case (vinsn_issue.vtype.vsew)
                EW8 : elem_idx[ 7:0] = masku_operand_vs1_seq[i* 8 +:  8];
                EW16: elem_idx[15:0] = masku_operand_vs1_seq[i*16 +: 16];
                EW32: elem_idx[31:0] = masku_operand_vs1_seq[i*32 +: 32];
                EW64: elem_idx[63:0] = masku_operand_vs1_seq[i*64 +: 64];
                default: ;
              endcase

              // Check if indices given by vs1 exist in vs
              if (((vcomp_vrgath_processed_element_vs2_cnt_q + i) < vinsn_issue.vl) && (elem_idx >= vcomp_vrgath_processed_element_vs2_cnt_q) && (elem_idx < (vcomp_vrgath_processed_element_vs2_cnt_q + total_input_elements))) begin
                unique case (vinsn_issue.vtype.vsew) // case statement gathers the elements from vs2 and puts them into vd
                  EW8 : vcomp_vrgath_result_d[i* 8 +:  8] = masku_operand_vs2_seq[(elem_idx%total_input_elements)* 8 +:  8];
                  EW16: vcomp_vrgath_result_d[i*16 +: 16] = masku_operand_vs2_seq[(elem_idx%total_input_elements)*16 +: 16];
                  EW32: vcomp_vrgath_result_d[i*32 +: 32] = masku_operand_vs2_seq[(elem_idx%total_input_elements)*32 +: 32];
                  EW64: vcomp_vrgath_result_d[i*64 +: 64] = masku_operand_vs2_seq[(elem_idx%total_input_elements)*64 +: 64];
                  default: ;
                endcase
                // vcomp_vrgath_result_element_cnt_d += 1'b1;// $display("at vs[%0d] = %0d, at %0dns", i, elem_idx, $time); // MOIMFELD: NOTE - Not needed for naive implementation where the core will ask for full pass though vs2 for ever vs1 slice
              end
            end

            // Update processed elements of vs2 counter (vcomp_vrgath_processed_element_vs2_cnt_d)
            vcomp_vrgath_processed_element_vs2_cnt_d += total_input_elements;

            // Check if whole vs2 has been processed (if so, we can be sure that the result vector is complete and ready to be written back to the vector register)
            vcomp_vrgath_result_valid = vcomp_vrgath_processed_element_vs2_cnt_d >= current_vlmax;

            // If vrgather result is valid, do:
            // - Compute byte enable signal (before updating vs1 pointer)
            // - Update processed elements of vs1 counter(vcomp_vrgath_processed_element_vs1_cnt_d)
            // - Acknowledge vs1 operand to receive next vs1 slice
            // - Reset processed elements of vs2 counter
            // - Ask for new operand (signal will be deasserted if instruction is finished, i.e. no more operands are needed)
            if (vcomp_vrgath_result_valid) begin
              // Compute byte enable signal
              for (int b = 0; b < ELENB * NrLanes; b++) begin
                automatic int mask_bit_idx = vcomp_vrgath_processed_element_vs1_cnt_d + (b >> vinsn_issue.vtype.vsew);
                if (masku_operand_m_seq[mask_bit_idx] || vinsn_issue.vm) begin
                  automatic int shuffle_byte = shuffle_index(b, NrLanes, vinsn_issue.vtype.vsew);
                  vrgath_be[shuffle_byte] = 1'b1;
                end
              end
              vcomp_vrgath_processed_element_vs1_cnt_d += total_input_elements;
              vcomp_vrgath_result_element_cnt_d = vcomp_vrgath_processed_element_vs1_cnt_d; // vcomp_vrgath_result_element_cnt_d is needed for address computation during result writeback
              masku_operand_vs1_ready_o = '1;
              vcomp_vrgath_processed_element_vs2_cnt_d = '0;
              masku_operand_req_o = '1;
            end

            if (vcomp_vrgath_processed_element_vs1_cnt_d >= vinsn_issue.vl) begin
              vrgather_finished = 1'b1;
              masku_operand_req_o = '0; // Do not request a new operand if instruction is finished
            end

          end
        end
        default: begin
          alu_result    = '0;
          alu_result_vm = '0;
        end
      endcase
    end

    // Shuffle result for masked instructions
    for (int b = 0; b < (NrLanes*StrbWidth); b++) begin
      automatic int shuffle_byte             = shuffle_index(b, NrLanes, vinsn_issue.vtype.vsew);
      alu_result_vm_seq[8*shuffle_byte +: 8] = alu_result_vm_m[8*b +: 8];
      vcomp_vrgath_result_shuffled[8*shuffle_byte +: 8] = vcomp_vrgath_result_d[8*b +: 8];
    end

    // alu_result propagation mux
    if (vinsn_issue.op inside {[VMSBF:VID]})
      alu_result = alu_result_vm_seq;

  end: p_mask_alu

  /////////////////
  //  Mask unit  //
  /////////////////

  // Vector instructions currently running
  logic [NrVInsn-1:0] vinsn_running_d, vinsn_running_q;

  // Interface with the main sequencer
  pe_resp_t pe_resp;

  // Effective MASKU stride in case of VSLIDEUP
  // MASKU receives chunks of 64 * NrLanes mask bits from the lanes
  // VSLIDEUP only needs the bits whose index >= than its stride
  // So, the operand requester does not send vl mask bits to MASKU
  // and trims all the unused 64 * NrLanes mask bits chunks
  // Therefore, the stride needs to be trimmed, too
  elen_t trimmed_stride;

  logic [NrLanes-1:0] fake_a_valid;
  logic last_incoming_a;
  logic unbalanced_a;

  // Control signals for better code-readability (this signals goes high if a result is valid and can be pushed to the result_queue)
  logic vreg_wb_valid;

  // Information about which is the target FU of the request
  assign masku_operand_fu = (vinsn_issue.op inside {[VMFEQ:VMFGE]}) ? MaskFUMFpu : MaskFUAlu;

  always_comb begin: p_masku
    // Maintain state
    vinsn_queue_d  = vinsn_queue_q;
    read_cnt_d     = read_cnt_q;
    issue_cnt_d    = issue_cnt_q;
    commit_cnt_d   = commit_cnt_q;

    mask_pnt_d     = mask_pnt_q;
    vrf_pnt_d      = vrf_pnt_q;

    vcpop_slice_cnt_d = vcpop_slice_cnt_q;
    popcount_d        = popcount_q;
    vfirst_count_d    = vfirst_count_q;

    mask_queue_d           = mask_queue_q;
    mask_queue_valid_d     = mask_queue_valid_q;
    mask_queue_write_pnt_d = mask_queue_write_pnt_q;
    mask_queue_read_pnt_d  = mask_queue_read_pnt_q;
    mask_queue_cnt_d       = mask_queue_cnt_q;

    result_queue_d           = result_queue_q;
    result_queue_valid_d     = result_queue_valid_q;
    result_queue_write_pnt_d = result_queue_write_pnt_q;
    result_queue_read_pnt_d  = result_queue_read_pnt_q;
    result_queue_cnt_d       = result_queue_cnt_q;

    result_final_gnt_d = result_final_gnt_q;

    trimmed_stride = pe_req_i.stride;

    // Vector instructions currently running
    vinsn_running_d = vinsn_running_q & pe_vinsn_running_i;

    // We are not ready, by default
    pe_resp                 = '0;
    masku_operand_a_ready_o = '0;
    masku_operand_m_ready_o = '0;

    // Inform the main sequencer if we are idle
    pe_req_ready_o = !vinsn_queue_full;

    // scalar path signals
    result_scalar_d       = result_scalar_o;
    result_scalar_valid_d = result_scalar_valid_o;

    // Balance the incoming valid
    unbalanced_a = (|commit_cnt_q[idx_width(NrLanes)-1:0] != 1'b0) ? 1'b1 : 1'b0;
    last_incoming_a = ((commit_cnt_q - vrf_pnt_q) < NrLanes) ? 1'b1 : 1'b0;
    fake_a_valid[0] = 1'b0;
    for (int unsigned i = 1; i < NrLanes; i++)
      if (i >= {1'b0, commit_cnt_q[idx_width(NrLanes)-1:0]})
        fake_a_valid[i] = last_incoming_a & unbalanced_a;
      else
        fake_a_valid = 1'b0;

    /////////////////////
    //  Mask Operands  //
    /////////////////////

    // Is there an instruction ready to be issued?
    if (vinsn_issue_valid && !(vd_scalar(vinsn_issue.op))) begin
      // Is there place in the mask queue to write the mask operands?
      // Did we receive the mask bits on the MaskM channel?
      if (!vinsn_issue.vm && !mask_queue_full && &masku_operand_m_valid_i && !(vinsn_issue.op inside {VRGATHER})) begin
        // Copy data from the mask operands into the mask queue
        for (int vrf_seq_byte = 0; vrf_seq_byte < NrLanes*StrbWidth; vrf_seq_byte++) begin
          // Map vrf_seq_byte to the corresponding byte in the VRF word.
          automatic int vrf_byte = shuffle_index(vrf_seq_byte, NrLanes, vinsn_issue.vtype.vsew);

          // At which lane, and what is the byte offset in that lane, of the byte vrf_byte?
          // NOTE: This does not work if the number of lanes is not a power of two.
          // If that is needed, the following two lines must be changed accordingly.
          automatic int vrf_lane   = vrf_byte >> $clog2(StrbWidth);
          automatic int vrf_offset = vrf_byte[idx_width(StrbWidth)-1:0];

          // The VRF pointer can be broken into a byte offset, and a bit offset
          automatic int vrf_pnt_byte_offset = mask_pnt_q >> $clog2(StrbWidth);
          automatic int vrf_pnt_bit_offset  = mask_pnt_q[idx_width(StrbWidth)-1:0];

          // A single bit from the mask operands can be used several times, depending on the eew.
          automatic int mask_seq_bit  = vrf_seq_byte >> int'(vinsn_issue.vtype.vsew);
          automatic int mask_seq_byte = (mask_seq_bit >> $clog2(StrbWidth)) + vrf_pnt_byte_offset;
          // Shuffle this source byte
          automatic int mask_byte     = shuffle_index(mask_seq_byte, NrLanes, vinsn_issue.eew_vmask);
          // Account for the bit offset
          automatic int mask_bit = (mask_byte << $clog2(StrbWidth)) +
            mask_seq_bit[idx_width(StrbWidth)-1:0] + vrf_pnt_bit_offset;

          // At which lane, and what is the bit offset in that lane, of the mask operand from
          // mask_seq_bit?
          automatic int mask_lane   = mask_bit >> idx_width(DataWidth);
          automatic int mask_offset = mask_bit[idx_width(DataWidth)-1:0];

          // Copy the mask operand
          mask_queue_d[mask_queue_write_pnt_q][vrf_lane][vrf_offset] =
            masku_operand_m[mask_lane][mask_offset];
        end

        // Account for the used operands
        mask_pnt_d += NrLanes * (1 << (int'(EW64) - vinsn_issue.vtype.vsew));

        // Increment result queue pointers and counters
        mask_queue_cnt_d += 1;
        if (mask_queue_write_pnt_q == MaskQueueDepth-1)
          mask_queue_write_pnt_d = '0;
        else
          mask_queue_write_pnt_d = mask_queue_write_pnt_q + 1;

        // Account for the operands that were issued
        read_cnt_d = read_cnt_q - NrLanes * (1 << (int'(EW64) - vinsn_issue.vtype.vsew));
        if (read_cnt_q < NrLanes * (1 << (int'(EW64) - vinsn_issue.vtype.vsew)))
          read_cnt_d = '0;

        // Trigger the request signal
        mask_queue_valid_d[mask_queue_write_pnt_q] = {NrLanes{1'b1}};

        // Are there lanes with no valid elements?
        // If so, mute their request signal
        if (read_cnt_q < NrLanes)
          mask_queue_valid_d[mask_queue_write_pnt_q] = (1 << read_cnt_q) - 1;

        // Consumed all valid bytes from the lane operands
        if (mask_pnt_d == NrLanes*64 || read_cnt_d == '0) begin
          // Request another beat
          masku_operand_m_ready_o = '1;
          // Reset the pointer
          mask_pnt_d              = '0;
        end
      end
    end

    //////////////////////////////
    // Calculate scalar results //
    //////////////////////////////

    // Is there an instruction ready to be issued?
    if (vinsn_issue_valid && vd_scalar(vinsn_issue.op)) begin
      if (&(masku_operand_a_valid_i | fake_a_valid) && (&masku_operand_m_valid_i || vinsn_issue.vm)) begin

        // increment slice counter
        vcpop_slice_cnt_d = vcpop_slice_cnt_q + 1'b1;

        // request new operand (by completing ready-valid handshake) once all slices have been processed
        masku_operand_a_ready_o = 1'b0;
        if (((vcpop_slice_cnt_q == N_SLICES_CPOP - 1) && vinsn_issue.op == VCPOP) ||
            ((vcpop_slice_cnt_q == N_SLICES_VFIRST-1) && vinsn_issue.op == VFIRST)) begin
          vcpop_slice_cnt_d       = '0;
          masku_operand_a_ready_o = masku_operand_a_valid_i;
          if (!vinsn_issue.vm) begin 
            masku_operand_m_ready_o = '1;
          end
        end

        // Account for the elements that were processed
        issue_cnt_d = issue_cnt_q - (W_CPOP/(8 << vinsn_issue.vtype.vsew));

        // abruptly stop processing elements if vl is reached
        if (iteration_count_d >= (vinsn_issue.vl/(W_CPOP)) || (!vfirst_empty && (vinsn_issue.op == VFIRST))) begin
          issue_cnt_d = '0;
          commit_cnt_d = '0;
          read_cnt_d ='0;
          masku_operand_a_ready_o = masku_operand_a_valid_i;
          if (!vinsn_issue.vm) begin 
            masku_operand_m_ready_o = '1;
          end
        end

        popcount_d     = popcount_q + popcount;
        vfirst_count_d = vfirst_count_q + vfirst_count;

        // if this is the last beat, commit the result to the scalar_result queue
        if ((iteration_count_d >= (vinsn_issue.vl/W_CPOP) && vinsn_issue.op == VCPOP) ||
            (iteration_count_d >= (vinsn_issue.vl/W_VFIRST) && vinsn_issue.op == VFIRST) ||
            (!vfirst_empty && (vinsn_issue.op == VFIRST))) begin
          result_scalar_d = (vinsn_issue.op == VCPOP) ? popcount_d : (vfirst_empty) ? -1 : vfirst_count_d;
          result_scalar_valid_d = '1;

          // Decrement the commit counter by the entire number of elements,
          // since we only commit one result for everything
          commit_cnt_d = '0;

          // reset vcpop slice counter, since instruction is finished
          vcpop_slice_cnt_d = '0;

          // acknowledge operand a
          masku_operand_a_ready_o = masku_operand_a_valid_i;
          if (!vinsn_issue.vm) begin 
            masku_operand_m_ready_o = '1;
          end
        end
      end
    end

    //////////////////////////////////
    //  Write results to the lanes  //
    //////////////////////////////////


    // By default, no writeback to the vector register
    // vreg_wb_valid = vinsn_issue_valid && !vd_scalar(vinsn_issue.op) && ;

    // MOIMFELD: Handling vcompress result writeback  -- TODO: CLEAN - lots of code repretition at the moment!!!!!

    // Is there an instruction ready to be issued?
    if (vinsn_issue_valid && !vd_scalar(vinsn_issue.op)) begin
      // This instruction executes on the Mask Unit
      if (vinsn_issue.vfu == VFU_MaskUnit) begin
        // Is there place in the result queue to write the results?
        // Did we receive the operands?
        if (!result_queue_full && ((&(masku_operand_a_valid_i | fake_a_valid) &&
            (!vinsn_issue.use_vd_op || &masku_operand_vs1_valid_i)) || vcomp_vrgath_result_valid)) begin
          if (!(vinsn_issue.op inside {VCOMPRESS, VRGATHER})) begin
            // How many elements are we committing in total?
            // Since we are committing bits instead of bytes, we carry out the following calculation
            // with ceil(vl/8) instead.
            automatic int element_cnt_all_lanes           = (ELENB * NrLanes) >> int'(vinsn_issue.vtype.vsew);
            // How many elements are remaining to be committed? Carry out the calculation with
            // ceil(issue_cnt/8).
            automatic int remaining_element_cnt_all_lanes = (issue_cnt_q + 7) / 8;
            remaining_element_cnt_all_lanes               = (remaining_element_cnt_all_lanes +
              (1 << int'(vinsn_issue.vtype.vsew)) - 1) >> int'(vinsn_issue.vtype.vsew);
            if (element_cnt_all_lanes > remaining_element_cnt_all_lanes)
              element_cnt_all_lanes = remaining_element_cnt_all_lanes;

            // Acknowledge the operands of this instruction.
            // At this stage, acknowledge only the first operand, "a", coming from the ALU/VMFpu.
            masku_operand_a_ready_o = masku_operand_a_valid_i;

            // Store the result in the operand queue
            for (int unsigned lane = 0; lane < NrLanes; lane++) begin
              // How many elements are we committing in this lane?
              automatic int element_cnt = element_cnt_all_lanes / NrLanes;
              if (lane < element_cnt_all_lanes[idx_width(NrLanes)-1:0])
                element_cnt += 1;

              result_queue_d[result_queue_write_pnt_q][lane] = '{
                wdata: result_queue_q[result_queue_write_pnt_q][lane].wdata | alu_result[lane],
                be   : (vinsn_issue.op inside {[VMSBF:VID]}) ? '1 : be(element_cnt, vinsn_issue.vtype.vsew),
                addr : (vinsn_issue.op inside {[VMSBF:VID]}) ? vaddr(vinsn_issue.vd, NrLanes) + ((vinsn_issue.vl - issue_cnt_q) >> (int'(EW64) - vinsn_issue.vtype.vsew)) : vaddr(vinsn_issue.vd, NrLanes) +
                  (((vinsn_issue.vl - issue_cnt_q) / NrLanes / DataWidth)),
                id : vinsn_issue.id
              };
            end
          end else begin // result_queue_d assignment for VRGATHER and VCOMPRESS
            for (int unsigned lane = 0; lane < NrLanes; lane++) begin
              automatic int element_cnt;
              automatic logic [63:0] base_addr;
              automatic logic [63:0] addr_offset;

              element_cnt = vcomp_vrgath_result_vec_elements / NrLanes;
              if ((vcomp_vrgath_result_vec_elements % NrLanes) != '0) begin
                element_cnt += '1;
              end

              base_addr = vaddr(vinsn_issue.vd, NrLanes);
              addr_offset = ((vcomp_vrgath_result_element_cnt_d - total_input_elements) >> (int'(EW64) - vinsn_issue.vtype.vsew)) / NrLanes; // MOIMFELD: TODO - CHECK IF IT IS WORKING FOR DIFFERENT SEW and add comment on how this is computed
              if ((vcomp_vrgath_result_element_cnt_d % total_input_elements) != '0) begin // MOIMFELD: TODO - Maybe change condition to "... < lane", this would really take care of fringe elements
                addr_offset += 1'b1;
              end
              // MOIMFELD: TODO - take care of fringe elements if result vector is not full --> do this in alu entry of VCOMPRESS and VRGATHER respectively and send signal to here that controls result writeback
              result_queue_d[result_queue_write_pnt_q][lane] = '{
                wdata: vcomp_vrgath_result_shuffled[ELEN * lane +: ELEN],
                be   : vinsn_issue.op inside {VCOMPRESS} ? be(element_cnt, vinsn_issue.vtype.vsew) : vrgath_be[ELENB*lane +: 8],
                addr : base_addr + addr_offset,
                id   : vinsn_issue.id
              };
            end
          end

          // Increment the VRF pointer
          if (vinsn_issue.op inside {[VMFEQ:VMSGTU], [VMSGT:VMSBC]}) begin
            vrf_pnt_d = vrf_pnt_q + (NrLanes << (int'(EW64) - vinsn_issue.vtype.vsew));

            // Filled-up a word, or finished execution
            if (vrf_pnt_d == DataWidth*NrLanes || vrf_pnt_d >= issue_cnt_q) begin
              result_queue_valid_d[result_queue_write_pnt_q] = {NrLanes{1'b1}};

              // Reset VRF pointer
              vrf_pnt_d = '0;

              // Increment result queue pointers and counters
              result_queue_cnt_d += 1;
              if (result_queue_write_pnt_q == ResultQueueDepth-1)
                result_queue_write_pnt_d = '0;
              else
                result_queue_write_pnt_d = result_queue_write_pnt_q + 1;

              // Account for the results that were issued
              issue_cnt_d = issue_cnt_q - NrLanes * DataWidth;
              if (issue_cnt_q < NrLanes * DataWidth)
                issue_cnt_d = '0;
            end
          end else if (vinsn_issue.op inside {[VMSBF:VID]}) begin
            result_queue_valid_d[result_queue_write_pnt_q] = {NrLanes{1'b1}};

            // Increment result queue pointers and counters
            result_queue_cnt_d += 1;
            if (result_queue_write_pnt_q == ResultQueueDepth-1)
              result_queue_write_pnt_d = '0;
            else
              result_queue_write_pnt_d = result_queue_write_pnt_q + 1;

            if (result_queue_read_pnt_q == ResultQueueDepth-1)
              result_queue_read_pnt_d = '0;
            else
              result_queue_read_pnt_d = result_queue_read_pnt_m;

            // Account for the results that were issued
            issue_cnt_d = issue_cnt_q - (1 << (int'(EW64) - vinsn_issue.vtype.vsew));
            if ((vinsn_issue.vl-issue_cnt_d)*4 >= vinsn_issue.vl)
              issue_cnt_d = '0;
          end else if (vinsn_issue.op inside {VCOMPRESS, VRGATHER}) begin
            if (vcomp_vrgath_result_valid) begin

              result_queue_valid_d[result_queue_write_pnt_q] = {NrLanes{1'b1}};
              // Increment result queue pointers and counters
              result_queue_cnt_d += 1;
              if (result_queue_write_pnt_q == ResultQueueDepth-1)
                result_queue_write_pnt_d = '0;
              else
                result_queue_write_pnt_d = result_queue_write_pnt_q + 1;
            end
          end else begin
            result_queue_valid_d[result_queue_write_pnt_q] = {NrLanes{1'b1}};

            // Increment result queue pointers and counters
            result_queue_cnt_d += 1;
            if (result_queue_write_pnt_q == ResultQueueDepth-1)
              result_queue_write_pnt_d = '0;
            else
              result_queue_write_pnt_d = result_queue_write_pnt_q + 1;

            // Account for the results that were issued
            issue_cnt_d = issue_cnt_q - NrLanes * DataWidth;
            if (issue_cnt_q < NrLanes * DataWidth)
              issue_cnt_d = '0;
          end
        end
      end
    end

    ///////////////////////////
    //// Masked Instruction ///
    ///////////////////////////
    if (vinsn_commit_valid && vinsn_commit.op inside {[VMSBF:VID]}) begin
      if (&masku_operand_a_valid_i && (&masku_operand_m_valid_i || vinsn_issue.vm)) begin
        // if this is the last beat, commit the result to the scalar_result queue
        commit_cnt_d = commit_cnt_q - (1 << (int'(EW64) - vinsn_commit.vtype.vsew));
        if ((vinsn_commit.vl-commit_cnt_d)*4 >= vinsn_commit.vl) begin
          commit_cnt_d = '0;
        end
      end
    end

    // Finished issuing results
    if (vinsn_issue_valid && (
          ( (vinsn_issue.vm || vinsn_issue.vfu == VFU_MaskUnit) && issue_cnt_d == '0) ||
          (!(vinsn_issue.vm || vinsn_issue.vfu == VFU_MaskUnit) && read_cnt_d == '0))) begin
      // Increment vector instruction queue pointers and counters
      vinsn_queue_d.issue_cnt -= 1;
    end

    /////////////////////////////////
    //  Send operands to the VFUs  //
    /////////////////////////////////

    for (int lane = 0; lane < NrLanes; lane++) begin: send_operand
      mask_valid_o[lane] = mask_queue_valid_q[mask_queue_read_pnt_q][lane];
      mask_o[lane]       = mask_queue_q[mask_queue_read_pnt_q][lane];
      // Received a grant from the VFUs.
      // The VLDU and the VSTU acknowledge all the operands at once.
      // Only accept the acknowledgement from the lanes if the current instruction is executing there.
      // Deactivate the request, but do not bump the pointers for now.
      if ((lane_mask_ready_i[lane] && mask_valid_o[lane] && vinsn_issue.vfu inside {VFU_Alu, VFU_MFpu, VFU_MaskUnit}) ||
           vldu_mask_ready_i || vstu_mask_ready_i || sldu_mask_ready_i) begin
        mask_queue_valid_d[mask_queue_read_pnt_q][lane] = 1'b0;
        mask_queue_d[mask_queue_read_pnt_q][lane]       = '0;
      end
    end: send_operand

    // Is this operand going to the lanes?
    mask_valid_lane_o = vinsn_issue.vfu inside {VFU_Alu, VFU_MFpu, VFU_MaskUnit};

    if (vd_scalar(vinsn_issue.op)) begin
      mask_valid_o = (vinsn_issue.vm) ? '0 : '1;
    end

    // All lanes accepted the VRF request
    if (!(|mask_queue_valid_d[mask_queue_read_pnt_q]))
      // There is something waiting to be written
      if (!mask_queue_empty) begin
        // Increment the read pointer
        if (mask_queue_read_pnt_q == MaskQueueDepth-1)
          mask_queue_read_pnt_d = 0;
        else
          mask_queue_read_pnt_d = mask_queue_read_pnt_q + 1;

        // Reset the queue
        mask_queue_d[mask_queue_read_pnt_q] = '0;

        // Decrement the counter of mask operands waiting to be used
        mask_queue_cnt_d -= 1;

        // Decrement the counter of remaining vector elements waiting to be used
        commit_cnt_d = commit_cnt_q - NrLanes * (1 << (int'(EW64) - vinsn_commit.vtype.vsew));
        if (commit_cnt_q < (NrLanes * (1 << (int'(EW64) - vinsn_commit.vtype.vsew))))
          commit_cnt_d = '0;
      end

    //////////////////////////////////
    //  Write results into the VRF  //
    //////////////////////////////////

    for (int lane = 0; lane < NrLanes; lane++) begin: result_write
      masku_result_req_o[lane]   = result_queue_valid_q[result_queue_read_pnt_q][lane];
      masku_result_addr_o[lane]  = result_queue_q[result_queue_read_pnt_q][lane].addr;
      masku_result_id_o[lane]    = result_queue_q[result_queue_read_pnt_q][lane].id;
      masku_result_wdata_o[lane] = result_queue_q[result_queue_read_pnt_q][lane].wdata;
      masku_result_be_o[lane]    = result_queue_q[result_queue_read_pnt_q][lane].be;

      // Update the final gnt vector
      result_final_gnt_d[lane] |= masku_result_final_gnt_i[lane];

      // Received a grant from the VRF.
      // Deactivate the request, but do not bump the pointers for now.
      if (masku_result_req_o[lane] && masku_result_gnt_i[lane]) begin
        result_queue_valid_d[result_queue_read_pnt_q][lane] = 1'b0;
        result_queue_d[result_queue_read_pnt_q][lane]       = '0;
        // Reset the final gnt vector since we are now waiting for another final gnt
        result_final_gnt_d[lane] = 1'b0;
      end
    end: result_write

    // All lanes accepted the VRF request
    if (!(|result_queue_valid_d[result_queue_read_pnt_q]) &&
      (&result_final_gnt_d || (commit_cnt_q > (NrLanes * DataWidth))))
      // There is something waiting to be written
      if (!result_queue_empty) begin
        // Increment the read pointer
        if (result_queue_read_pnt_q == ResultQueueDepth-1)
          result_queue_read_pnt_d = 0;
        else
          result_queue_read_pnt_d = result_queue_read_pnt_q + 1;

        // Decrement the counter of results waiting to be written
        result_queue_cnt_d -= 1;

        // Reset the queue
        result_queue_d[result_queue_read_pnt_q] = '0;

        // Decrement the counter of remaining vector elements waiting to be written
        if (!(vinsn_issue.op inside {VCOMPRESS, VRGATHER})) begin
          commit_cnt_d = commit_cnt_q - NrLanes * DataWidth;
          if (commit_cnt_q < (NrLanes * DataWidth))
            commit_cnt_d = '0;
        end
      end

    // vcompress does not have a fixed number of results. Therefore, the commit counter
    // cannot be reused as an indicator for when the instruciton is done. For this reason
    // the signal 'vcompress_finished' is introduced.
    if ((vinsn_issue.op inside {VCOMPRESS, VRGATHER}) && (vcompress_finished || vrgather_finished)) begin
      commit_cnt_d = '0;
      issue_cnt_d  = '0;
      read_cnt_d   = '0;
    end

    ///////////////////////////
    // Commit scalar results //
    ///////////////////////////

    // The scalar result has been sent to and acknowledged by the dispatcher
    if (vinsn_commit.op inside {[VCPOP:VFIRST]} && result_scalar_valid_o == 1) begin

      // reset result_scalar
      result_scalar_d       = '0;
      result_scalar_valid_d = '0;

      // reset the popcount and vfirst_count
      popcount_d     = '0;
      vfirst_count_d = '0;
    end

    // Finished committing the results of a vector instruction
    // Some instructions forward operands to the lanes before writing the VRF
    // In this case, wait for the lanes to be written
    if (vinsn_commit_valid && commit_cnt_d == '0 &&
      (!(vinsn_commit.op inside {[VMFEQ:VID], [VMSGT:VMSBC]}) || &result_final_gnt_d)) begin
      // Mark the vector instruction as being done
      pe_resp.vinsn_done[vinsn_commit.id] = 1'b1;

      // Update the commit counters and pointers
      vinsn_queue_d.commit_cnt -= 1;
    end

    //////////////////////////////
    //  Accept new instruction  //
    //////////////////////////////

    // Trim the slide stride if it is higher than NrLanes * 64
    // and we have a VSLIDEUP, as the mask bits with index lower than
    // this stride are not used and therefore not sent to the MASKU
    if (pe_req_i.stride >= NrLanes * 64)
      trimmed_stride = pe_req_i.stride - ((pe_req_i.stride >> NrLanes * 64) << NrLanes * 64);

    if (!vinsn_queue_full && pe_req_valid_i && !vinsn_running_q[pe_req_i.id] &&
        (!pe_req_i.vm || pe_req_i.vfu == VFU_MaskUnit)) begin
      vinsn_queue_d.vinsn[0]       = pe_req_i;
      vinsn_running_d[pe_req_i.id] = 1'b1;

      // Initialize counters
      if (vinsn_queue_d.issue_cnt == '0) begin
        issue_cnt_d = pe_req_i.vl;
        read_cnt_d  = pe_req_i.vl;

        // Trim skipped words
        if (pe_req_i.op == VSLIDEUP) begin
          issue_cnt_d -= vlen_t'(trimmed_stride);
          case (pe_req_i.vtype.vsew)
            EW8:  begin
              read_cnt_d -= (vlen_t'(trimmed_stride) >> $clog2(NrLanes << 3)) << $clog2(NrLanes << 3);
              mask_pnt_d  = (vlen_t'(trimmed_stride) >> $clog2(NrLanes << 3)) << $clog2(NrLanes << 3);
            end
            EW16: begin
              read_cnt_d -= (vlen_t'(trimmed_stride) >> $clog2(NrLanes << 2)) << $clog2(NrLanes << 2);
              mask_pnt_d  = (vlen_t'(trimmed_stride) >> $clog2(NrLanes << 2)) << $clog2(NrLanes << 2);
            end
            EW32: begin
              read_cnt_d -= (vlen_t'(trimmed_stride) >> $clog2(NrLanes << 1)) << $clog2(NrLanes << 1);
              mask_pnt_d  = (vlen_t'(trimmed_stride) >> $clog2(NrLanes << 1)) << $clog2(NrLanes << 1);
            end
            EW64: begin
              read_cnt_d -= (vlen_t'(trimmed_stride) >> $clog2(NrLanes)) << $clog2(NrLanes);
              mask_pnt_d  = (vlen_t'(trimmed_stride) >> $clog2(NrLanes)) << $clog2(NrLanes);
            end
            default:;
          endcase
        end

        // Reset the final grant vector
        // Be aware: this works only if the insn queue length is 1

        result_final_gnt_d = '0;
      end
      if (vinsn_queue_d.commit_cnt == '0) begin
        commit_cnt_d = pe_req_i.vl;
        // Trim skipped words
        if (pe_req_i.op == VSLIDEUP)
          commit_cnt_d -= vlen_t'(trimmed_stride);
      end

      // Bump pointers and counters of the vector instruction queue
      vinsn_queue_d.issue_cnt += 1;
      vinsn_queue_d.commit_cnt += 1;
    end
  end: p_masku

  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      vinsn_running_q    <= '0;
      read_cnt_q         <= '0;
      issue_cnt_q        <= '0;
      commit_cnt_q       <= '0;
      vrf_pnt_q          <= '0;
      mask_pnt_q         <= '0;
      pe_resp_o          <= '0;
      result_final_gnt_q <= '0;
      vcpop_slice_cnt_q  <= '0;
      popcount_q         <= '0;
      vfirst_count_q     <= '0;
      vcomp_vrgath_result_element_cnt_q <= '0;
      vcomp_vrgath_processed_element_vs1_cnt_q <= '0;
      vcomp_vrgath_processed_element_vs2_cnt_q <= '0;
      vcomp_vrgath_result_q <= '0;
    end else begin
      vinsn_running_q    <= vinsn_running_d;
      read_cnt_q         <= read_cnt_d;
      issue_cnt_q        <= issue_cnt_d;
      commit_cnt_q       <= commit_cnt_d;
      vrf_pnt_q          <= vrf_pnt_d;
      mask_pnt_q         <= mask_pnt_d;
      pe_resp_o          <= pe_resp;
      result_final_gnt_q <= result_final_gnt_d;
      vcpop_slice_cnt_q  <= vcpop_slice_cnt_d;
      popcount_q         <= popcount_d;
      vfirst_count_q     <= vfirst_count_d;
      vcomp_vrgath_result_element_cnt_q    <= vcomp_vrgath_result_element_cnt_d;
      vcomp_vrgath_processed_element_vs1_cnt_q <= vcomp_vrgath_processed_element_vs1_cnt_d;
      vcomp_vrgath_processed_element_vs2_cnt_q <= vcomp_vrgath_processed_element_vs2_cnt_d;
      vcomp_vrgath_result_q <= vcomp_vrgath_result_d;
    end
  end

endmodule : masku
