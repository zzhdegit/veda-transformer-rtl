# Hardware Stage H9 Performance Root Cause

Status: matched RTL audit complete for single-head Attention.

The matched RTL A/B shows that H9 has fixed stream setup overhead on short
sequences, but wins on longer sequences through two mechanisms:

- QK/SFU and SFU/sV overlap remove staged SFU serialization.
- Native mapping avoids the Stage 8 staged paper adapter's low-lane D_HEAD
  tiling cost.

## D_HEAD=8 Delta Against Paper Staged RTL

Positive cycle delta means H9 is faster than staged. Negative means H9 is
slower.

| Cause | Seq 8 | Seq 16 | Seq 32 | Percent Basis |
|---|---:|---:|---:|---:|
| Total cycle delta | 42 | 194 | 498 | total staged cycles |
| QK issue/array bookkeeping overhead | -88 | -64 | -16 | H9 qk minus staged qk |
| Extra scale/reduction bookkeeping | -48 | -48 | -48 | H9 scale+reduce minus staged |
| Reduction finalize saving | 11 | 11 | 11 | staged minus H9 |
| Normalization overlap/saving | 15 | 31 | 63 | staged norm minus H9 norm |
| sV overlap/saving | 9 | 25 | 57 | staged sv minus H9 sv |
| Staged PE stall removed | 522 | 1042 | 2082 | staged pe_stall minus H9 pe_stall |
| Extra SFU wait in H9 | -7 | -23 | -55 | H9 sfu_stall minus staged |
| Controller/output stalls | 0 | 0 | 0 | output_stall delta |

At D_HEAD=8, H9 breaks even at seq8 and clears the seq16/32 acceptance points.
The remaining H9 overhead is the stream controller and QK/SFU handshake
bookkeeping; the dominant saving is removal of staged PE stalls.

## D_HEAD=16 And D_HEAD=64

| D_HEAD | Seq | Staged | H9 | Improvement | Main cause |
|---:|---:|---:|---:|---:|---|
| 16 | 16 | 2472 | 1171 | 52.6% | native mapping plus overlap |
| 16 | 32 | 4920 | 2211 | 55.1% | native mapping plus overlap |
| 64 | 16 | 9126 | 1183 | 87.0% | full-array D_HEAD mapping |
| 64 | 32 | 18198 | 2223 | 87.8% | full-array D_HEAD mapping |

For D_HEAD=16 and D_HEAD=64, the Stage 8 staged paper path repeatedly tiles
through the low-lane adapter. H9's native group/row/column mapping uses both
groups and the paper hierarchy directly, so the improvement is much larger than
the overlap-only gain.

## Short Sequence Cost

| D_HEAD | Seq | Staged | H9 | Result |
|---:|---:|---:|---:|---|
| 8 | 1 | 91 | 194 | H9 fixed overhead dominates |
| 8 | 2 | 187 | 259 | H9 fixed overhead dominates |
| 16 | 1 | 165 | 196 | H9 fixed overhead dominates |

Seq1/2 overhead is expected and remains explicitly reported. It is not hidden
or used to claim universal speedup.
