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

module masku_operands import ara_pkg::*; import rvv_pkg::*; #(
    parameter int unsigned NrLanes = 0
  ) (
    // Control logic
    input masku_fu_e masku_fu_i,

    // Operands coming from lanes
    input  elen_t [NrLanes-1:0][NrMaskFUnits+2-1:0] masku_operands_i,

    // Operands prepared for masku execution
    output elen_t [NrLanes-1:0] masku_operand_a_o, // ALU/FPU result
    output elen_t [NrLanes-1:0] masku_operand_b_o, // Previous value of the destination vector register
    output elen_t [NrLanes-1:0] masku_operand_m_o // Mask
  );

  // Extract operands from input (input comes in shuffled form)
  for (genvar lane = 0; lane < NrLanes; lane++) begin
    assign masku_operand_a_o[lane] = masku_operands_i[lane][2 + masku_fu_i];
    assign masku_operand_b_o[lane] = masku_operands_i[lane][1];
    assign masku_operand_m_o[lane] = masku_operands_i[lane][0];
  end

endmodule : masku_operands
