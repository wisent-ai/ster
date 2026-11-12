#!/bin/sh
# Real test of how choosing the device and the precision fails
# (`ster.runtime.device`), through `ster generate` on the four-layer
# fixture. An unknown word and bf16 on the cpu are usage refusals (exit 2);
# a device this binary was built without exits 1: cuda on a macOS build,
# metal on any other.
#
# Usage: tests/runtime/device.sh   (STER selects the binary, default target/debug/ster)
AREA=device
. "$(dirname "$0")/harness.sh"

an_unknown_device_is_refused() {
  generate 2 "$MODEL" "describe the sea ." --device tpu
  said "unknown device \"tpu\"; expected cpu, metal, or cuda"
  passed an_unknown_device_is_refused
}

a_device_this_build_lacks_is_refused() {
  case "$(uname -s)" in Darwin) lacking=cuda ;; *) lacking=metal ;; esac
  generate 1 "$MODEL" "describe the sea ." --device "$lacking"
  said "Error: opening the $lacking device: this Ster binary was built without the $lacking feature; observed: --device $lacking"
  passed a_device_this_build_lacks_is_refused
}

an_unknown_precision_is_refused() {
  generate 2 "$MODEL" "describe the sea ." --precision f64
  said "unknown precision \"f64\"; expected f32, f16, or bf16"
  passed an_unknown_precision_is_refused
}

bf16_on_the_cpu_is_refused() {
  generate 2 "$MODEL" "describe the sea ." --precision bf16 --device cpu
  said "Error: mapping the weights of $MODEL at bf16 on the cpu: bf16 has no CPU matmul kernel in this Candle build, so the run would load the whole checkpoint and then fail at its first projection; observed: --precision bf16 --device cpu (cpu is the device when --device is left out); nothing of $MODEL was read; fix: use --precision f16 for half precision on cpu, or --device metal for bf16"
  passed bf16_on_the_cpu_is_refused
}

an_unknown_device_is_refused
a_device_this_build_lacks_is_refused
an_unknown_precision_is_refused
bf16_on_the_cpu_is_refused

touch "$ROOT/passed"
echo "PASS" >>"$REPORT"
echo "PASS: $REPORT"
