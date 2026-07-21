# Ensemble-composition screening — generated tables

Replica-vs-published validation (binary, non-stacked): exact=365, close<0.005=1, off≥0.005=3, mean |Δp|=0.00030. Total CI-bearing comparisons reported: 48.


## fall_mean3  (method=mean, n_scored=41, types={'discrete': 4, 'multiple_choice': 6, 'numeric': 9, 'binary': 22}, stacked_published=0, anon_recovered=0)

Families: anthropic=['claude-sonnet-4']; openai-gpt5=['gpt-5']; openai-o3=['o3']

| comparison | n_q | mean Δlog vs replica | 95% CI | ΔBrier (binary) | Brier CI |
|---|---|---|---|---|---|
| drop_anthropic | 41 | -1.68 | [-9.77, +4.14] | +0.0104 (n=22) | [-0.0168, +0.0448] |
| drop_openai-gpt5 | 41 | -2.09 | [-6.62, +2.08] | +0.0053 (n=22) | [-0.0152, +0.0271] |
| drop_openai-o3 | 41 | -0.89 | [-7.36, +3.90] | -0.0082 (n=22) | [-0.0214, +0.0045] |
| top3_families_LOQO | 0 | — skipped (n<8) | | | |

Per-member mean log score (n≥5):

| model | n | mean log |
|---|---|---|
| gpt-5 | 41 | +32.63 |
| o3 | 41 | +31.12 |
| claude-sonnet-4 | 41 | +26.11 |

## fall_5m  (method=median, n_scored=36, types={'discrete': 2, 'multiple_choice': 3, 'binary': 19, 'numeric': 12}, stacked_published=0, anon_recovered=0)

Families: anthropic=['claude-sonnet-4']; kimi=['kimi-k2-0905']; openai-gpt5=['gpt-5']; openai-o3=['o3']; qwen=['qwen3-235b-a22b-thinking-2507']

| comparison | n_q | mean Δlog vs replica | 95% CI | ΔBrier (binary) | Brier CI |
|---|---|---|---|---|---|
| drop_anthropic | 36 | +1.03 | [-1.86, +4.06] | +0.0025 (n=19) | [-0.0099, +0.0152] |
| drop_kimi | 35 | -0.82 | [-3.37, +2.13] | +0.0017 (n=18) | [-0.0096, +0.0113] |
| drop_openai-gpt5 | 36 | -6.42 | [-13.25, -1.00] | +0.0123 (n=19) | [+0.0013, +0.0261] |
| drop_openai-o3 | 36 | -2.25 | [-7.18, +1.78] | +0.0113 (n=19) | [+0.0001, +0.0259] |
| drop_qwen | 36 | -2.67 | [-7.72, +1.15] | +0.0080 (n=19) | [-0.0030, +0.0238] |
| top3_families_LOQO | 36 | +1.60 | [-2.62, +6.02] | +0.0033 (n=19) | [-0.0182, +0.0226] |

Per-member mean log score (n≥5):

| model | n | mean log |
|---|---|---|
| gpt-5 | 36 | +34.91 |
| o3 | 36 | +29.35 |
| qwen3-235b-a22b-thinking-2507 | 36 | +23.04 |
| claude-sonnet-4 | 36 | +17.95 |
| kimi-k2-0905 | 35 | +10.99 |

## fall_6m  (method=median, n_scored=277, types={'multiple_choice': 18, 'numeric': 74, 'discrete': 18, 'binary': 167}, stacked_published=0, anon_recovered=18)

Families: anthropic=['claude-sonnet-4', 'claude-sonnet-4.5']; grok=['grok-4-fast', 'grok-4.1-fast']; kimi=['kimi-k2-0905']; openai-gpt5=['gpt-5', 'gpt-5.1', 'unattributed:openai-gpt5']; openai-o3=['o3']; qwen=['qwen3-235b-a22b-thinking-2507']

| comparison | n_q | mean Δlog vs replica | 95% CI | ΔBrier (binary) | Brier CI |
|---|---|---|---|---|---|
| drop_anthropic | 274 | -1.34 | [-3.01, +0.07] | +0.0006 (n=167) | [-0.0036, +0.0053] |
| drop_grok | 270 | +0.36 | [-1.17, +1.71] | -0.0036 (n=160) | [-0.0077, -0.0001] |
| drop_kimi | 275 | +0.63 | [-0.48, +1.73] | -0.0029 (n=167) | [-0.0067, +0.0005] |
| drop_openai-gpt5 | 275 | -1.75 | [-3.08, -0.53] | +0.0059 (n=166) | [+0.0017, +0.0113] |
| drop_openai-o3 | 277 | -1.28 | [-2.62, -0.02] | +0.0058 (n=167) | [+0.0013, +0.0112] |
| drop_qwen | 275 | -0.09 | [-1.69, +1.31] | +0.0003 (n=166) | [-0.0042, +0.0049] |
| top3_families_LOQO | 277 | +1.13 | [-1.29, +3.62] | -0.0027 (n=167) | [-0.0089, +0.0053] |

Per-member mean log score (n≥5):

| model | n | mean log |
|---|---|---|
| unattributed:openai-gpt5 | 18 | +96.19 |
| claude-sonnet-4.5 | 232 | +57.48 |
| gpt-5 | 208 | +56.49 |
| gpt-5.1 | 49 | +55.03 |
| o3 | 277 | +54.98 |
| kimi-k2-0905 | 273 | +47.24 |
| grok-4-fast | 251 | +45.68 |
| qwen3-235b-a22b-thinking-2507 | 275 | +45.26 |
| claude-sonnet-4 | 42 | +31.22 |
| grok-4.1-fast | 19 | +29.19 |

## spring_5m_a  (method=median, n_scored=69, types={'numeric': 27, 'multiple_choice': 6, 'discrete': 7, 'binary': 29}, stacked_published=0, anon_recovered=0)

Families: anthropic=['claude-opus-4.5']; gemini=['gemini-3-flash-preview', 'gemini-3-pro-preview']; openai-gpt5=['gpt-5', 'gpt-5.2']

| comparison | n_q | mean Δlog vs replica | 95% CI | ΔBrier (binary) | Brier CI |
|---|---|---|---|---|---|
| drop_anthropic | 66 | +1.53 | [-1.83, +5.14] | +0.0005 (n=29) | [-0.0162, +0.0149] |
| drop_gemini | 69 | -2.13 | [-6.50, +2.88] | -0.0056 (n=29) | [-0.0356, +0.0188] |
| drop_openai-gpt5 | 69 | +1.55 | [-3.12, +6.08] | -0.0024 (n=29) | [-0.0276, +0.0253] |
| top3_families_LOQO | 0 | — skipped (n<8) | | | |

Per-member mean log score (n≥5):

| model | n | mean log |
|---|---|---|
| gemini-3-pro-preview | 69 | +46.07 |
| claude-opus-4.5 | 66 | +43.60 |
| gpt-5 | 69 | +43.07 |
| gemini-3-flash-preview | 68 | +42.81 |
| gpt-5.2 | 69 | +41.13 |

## spring_trans  (method=median, n_scored=37, types={'binary': 26, 'multiple_choice': 2, 'discrete': 4, 'numeric': 5}, stacked_published=0, anon_recovered=0)

Families: anthropic=['claude-opus-4.5', 'claude-opus-4.6']; gemini=['gemini-3-flash-preview', 'gemini-3-pro-preview', 'gemini-3.1-pro-preview']; openai-gpt5=['gpt-5', 'gpt-5.1', 'gpt-5.2']

| comparison | n_q | mean Δlog vs replica | 95% CI | ΔBrier (binary) | Brier CI |
|---|---|---|---|---|---|
| drop_anthropic | 37 | +4.18 | [-1.10, +10.36] | -0.0181 (n=26) | [-0.0398, -0.0011] |
| drop_gemini | 37 | -4.12 | [-9.00, -0.75] | +0.0056 (n=26) | [-0.0008, +0.0130] |
| drop_openai-gpt5 | 37 | -8.32 | [-19.24, +0.50] | +0.0261 (n=26) | [-0.0055, +0.0680] |
| top3_families_LOQO | 0 | — skipped (n<8) | | | |

Per-member mean log score (n≥5):

| model | n | mean log |
|---|---|---|
| gemini-3-flash-preview | 27 | +43.25 |
| gpt-5.2 | 34 | +41.79 |
| gpt-5.1 | 8 | +40.21 |
| gemini-3-pro-preview | 35 | +31.91 |
| gpt-5 | 27 | +27.29 |
| claude-opus-4.6 | 37 | +15.58 |
| claude-opus-4.5 | 10 | +8.85 |

## spring_5m_b  (method=median, n_scored=138, types={'multiple_choice': 24, 'binary': 87, 'discrete': 10, 'numeric': 17}, stacked_published=0, anon_recovered=19)

Families: anthropic=['claude-opus-4.5', 'claude-opus-4.6']; gemini=['gemini-3.1-pro-preview']; openai-gpt5=['gpt-5.1', 'gpt-5.2']

| comparison | n_q | mean Δlog vs replica | 95% CI | ΔBrier (binary) | Brier CI |
|---|---|---|---|---|---|
| drop_anthropic | 137 | +3.29 | [+0.25, +6.47] | -0.0164 (n=86) | [-0.0294, -0.0048] |
| drop_gemini | 138 | -2.05 | [-3.72, -0.41] | +0.0058 (n=87) | [-0.0011, +0.0130] |
| drop_openai-gpt5 | 137 | -5.18 | [-9.13, -1.90] | +0.0231 (n=86) | [+0.0108, +0.0364] |
| top3_families_LOQO | 0 | — skipped (n<8) | | | |

Per-member mean log score (n≥5):

| model | n | mean log |
|---|---|---|
| gpt-5.1 | 137 | +34.33 |
| gpt-5.2 | 136 | +33.75 |
| gemini-3.1-pro-preview | 138 | +30.75 |
| claude-opus-4.5 | 137 | +22.26 |
| claude-opus-4.6 | 135 | +21.12 |

## spring_6m  (method=median, n_scored=13, types={'multiple_choice': 4, 'discrete': 2, 'binary': 7}, stacked_published=0, anon_recovered=0)

Families: anthropic=['claude-opus-4.5', 'claude-opus-4.6']; gemini=['gemini-3.1-pro-preview']; grok=['grok-4.1-fast']; openai-gpt5=['gpt-5.1', 'gpt-5.2', 'gpt-5.4']

| comparison | n_q | mean Δlog vs replica | 95% CI | ΔBrier (binary) | Brier CI |
|---|---|---|---|---|---|
| drop_anthropic | 13 | +0.29 | [-5.36, +5.64] | +0.0027 (n=7) | [-0.0210, +0.0326] |
| drop_gemini | 13 | -0.05 | [-1.65, +2.08] | -0.0004 (n=7) | [-0.0061, +0.0053] |
| drop_grok | 12 | +1.07 | [-0.73, +3.29] | -0.0004 (n=7) | [-0.0061, +0.0053] |
| drop_openai-gpt5 | 13 | -0.18 | [-6.57, +4.91] | +0.0019 (n=7) | [-0.0252, +0.0372] |
| drop_grok+gemini | 13 | +2.06 | [-2.13, +8.00] | -0.0046 (n=7) | [-0.0204, +0.0094] |
| top3_families_LOQO | 13 | +0.98 | [-0.64, +3.01] | -0.0004 (n=7) | [-0.0061, +0.0050] |

Per-member mean log score (n≥5):

| model | n | mean log |
|---|---|---|
| claude-opus-4.6 | 13 | +29.71 |
| gpt-5.2 | 13 | +29.13 |
| gemini-3.1-pro-preview | 13 | +25.60 |
| claude-opus-4.5 | 13 | +25.49 |
| gpt-5.4 | 12 | +24.00 |
| grok-4.1-fast | 12 | +18.59 |

## summer_6m  (method=median, n_scored=45, types={'binary': 15, 'multiple_choice': 6, 'numeric': 22, 'discrete': 2}, stacked_published=4, anon_recovered=0)

Families: anthropic=['claude-opus-4.6', 'claude-opus-4.7', 'claude-opus-4.8']; gemini=['gemini-3.1-pro-preview']; grok=['grok-4.3']; openai-gpt5=['gpt-5.4', 'gpt-5.5']

| comparison | n_q | mean Δlog vs replica | 95% CI | ΔBrier (binary) | Brier CI |
|---|---|---|---|---|---|
| drop_anthropic | 45 | -1.71 | [-3.85, +0.65] | +0.0047 (n=15) | [-0.0165, +0.0212] |
| drop_gemini | 42 | -1.40 | [-3.85, +1.20] | +0.0127 (n=14) | [+0.0039, +0.0238] |
| drop_grok | 45 | -0.67 | [-5.28, +2.74] | -0.0069 (n=15) | [-0.0172, +0.0019] |
| drop_openai-gpt5 | 45 | +0.96 | [-2.40, +3.76] | -0.0026 (n=15) | [-0.0209, +0.0225] |
| drop_grok+gemini | 45 | -0.17 | [-2.24, +1.83] | +0.0020 (n=15) | [-0.0056, +0.0082] |
| top3_families_LOQO | 45 | -0.67 | [-5.29, +2.79] | -0.0069 (n=15) | [-0.0171, +0.0022] |
| unstacked_only/drop_anthropic | 41 | -1.06 | [-3.14, +1.20] | -0.0026 (n=12) | [-0.0251, +0.0121] |
| unstacked_only/drop_gemini | 38 | -0.88 | [-3.40, +1.98] | +0.0093 (n=11) | [+0.0018, +0.0182] |
| unstacked_only/drop_grok | 41 | -1.26 | [-6.41, +2.37] | -0.0031 (n=12) | [-0.0128, +0.0053] |
| unstacked_only/drop_openai-gpt5 | 41 | +0.12 | [-3.52, +3.06] | +0.0047 (n=12) | [-0.0130, +0.0332] |

Per-member mean log score (n≥5):

| model | n | mean log |
|---|---|---|
| claude-opus-4.8 | 20 | +50.52 |
| gemini-3.1-pro-preview | 42 | +48.43 |
| gpt-5.5 | 45 | +39.44 |
| claude-opus-4.6 | 45 | +38.51 |
| claude-opus-4.7 | 25 | +34.41 |
| gpt-5.4 | 43 | +33.70 |
| grok-4.3 | 45 | +27.26 |

## Pooled per-family LOO (descriptive — per-era tables are primary)

| comparison | n_q | eras | mean Δlog | 95% CI |
|---|---|---|---|---|
| drop_grok | 327 | fall_6m,spring_6m,summer_6m | +0.24 | [-1.13, +1.48] |
| drop_gemini | 299 | spring_5m_a,spring_trans,spring_5m_b,spring_6m,summer_6m | -2.14 | [-3.62, -0.62] |
| drop_anthropic | 649 | fall_mean3,fall_5m,fall_6m,spring_5m_a,spring_trans,spring_5m_b,spring_6m,summer_6m | +0.36 | [-0.84, +1.50] |
| drop_openai-gpt5 | 653 | fall_mean3,fall_5m,fall_6m,spring_5m_a,spring_trans,spring_5m_b,spring_6m,summer_6m | -2.56 | [-3.90, -1.25] |
| drop_openai-o3 | 354 | fall_mean3,fall_5m,fall_6m | -1.33 | [-2.70, -0.10] |
| drop_kimi | 310 | fall_5m,fall_6m | +0.47 | [-0.59, +1.50] |
| drop_qwen | 311 | fall_5m,fall_6m | -0.39 | [-1.78, +0.92] |

## Drop counts

- fewer_than_2_attributed_members: 7
- no_members_binary: 15
- no_members_multiple_choice: 2
- no_members_numeric: 14
- numeric_member_pchip_failed: 2
- numeric_member_too_few_percentiles: 3
- stacked_binary_unrecoverable: 12
- stacked_multiple_choice_unrecoverable: 2
- stacked_numeric_unrecoverable: 4
