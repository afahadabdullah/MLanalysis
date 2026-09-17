# Parked insertion data sources (upper air / moisture)

**Status:** superseded by `PROJECT_PLAN_LEAN.md` revision 6, whose core test is mesonet 2 m T. These candidates are kept for possible follow-on experiments.

| Variable → GraphCast input | Source | In MERRA-2? | Note |
|---|---|---|---|
| T profile → `t` | IGRA2 radiosondes | Yes | Textbook imbalance test (T without z/winds) |
| q profile → `q` | IGRA2 dewpoint | Yes (*confirm*) | Moist imbalance |
| Column water → scale `q` | RSS AMSR2 TPW (ocean) | AMSR2 not listed in McCarty et al. 2016 (*confirm*; also check ERA5) | Candidate "new information" moisture |
| 10 m winds → `10u`, `10v` | ASCAT L2 | Yes | Avoid CCMP v3.1: it uses an ERA5 background |
| T/q profiles | COSMIC-2 wetPf2, AIRS v7 | Bending angle / radiances yes | Retrieval priors complicate independence |
| 6 h precipitation → `tp` | IMERG | No | Input channel only, not prognostic; low priority |

Source for MERRA-2 observing system: McCarty et al. (2016), https://gmao.gsfc.nasa.gov/pubs/docs/McCarty885.pdf
