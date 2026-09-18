import sys
sys.path.insert(0, "<home>/qwen38-flash-next-mi50/harness")
import results
r = results.make_row("REHEARSAL", 0, 0.0, 900,
    notes="day-one dry run: mechanics validated", shape_profile="microbench",
    graphs="off", dtype="float16", ple_source="none", ple_offload=False,
    tokens_per_second=None, steps_per_second=None)
results.append_row(r)
print("REHEARSAL row written (schema-valid v2)")
