# ML-M3 RTL Smoke Results

- H8/D8/D_MODEL64/D_FFN256 compile/elaborate: PASS
- H8 staged one-token: FAIL
- H9 interleaved one-token: FAIL
- First observable difference: `CHECK_FAIL layer token=0 dim=1 got=3d4a2576 expected=3d4a2572`
- H8/H9 partial capture SHA match: True
- H8 partial capture SHA256: `5adbf7b5ef5e5fbff1a767e271d852ab711ec9d829a9f7fe9125288901d4f3be`
- H9 partial capture SHA256: `5adbf7b5ef5e5fbff1a767e271d852ab711ec9d829a9f7fe9125288901d4f3be`

Because one-token smoke failed, ML-M3 did not enter multi-token RTL co-simulation.
