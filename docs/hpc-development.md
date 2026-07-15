# HPC development

## Current status

Phase 01 contains non-production Slurm templates only. No Discoverer, EuroHPC, multi-node, or multi-GPU benchmark has been performed or claimed.

## Purpose of templates

The templates establish where future scheduler directives, environment activation, process layout, and diagnostic commands will live. Values must be verified against the allocated system before use.

## Evidence required before a resource request

- exact Git commit and environment;
- fixed benchmark workload;
- node, process, thread, and GPU layout;
- wall time and useful throughput;
- memory high-water mark;
- I/O volume and wait time;
- communication time;
- numerical equivalence;
- strong- or weak-scaling efficiency;
- failed-run record.

Resource requests must be derived from measured pilot performance rather than theoretical assumptions.
