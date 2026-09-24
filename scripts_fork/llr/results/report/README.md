# NOTE: new file in this fork -- not in upstream NVlabs/alpamayo-recipes.

# The Phase-0 report

`llr_report.html` is the write-up of the discrete-token-head results: open it in a browser, it
is self-contained (all figures inlined, no external assets, works offline).

It was previously published as a shared page; that link is gone and this file is now the copy
of record. It is **not** auto-generated -- if a number in `results/phase0_llr_action_direction.md`
changes, this page does not follow, so treat the markdown as the source of truth for numbers and
this page as the presentation of them.

Known loose end, worth fixing if the page is ever revised: the "Every effect as a share of the
separation" table still carries a row called *sharpening from reasoning* (+0.0014, n.s.). That is
`separation(gold CoC) - separation(empty CoC)`, i.e. how much of reasoning's benefit is specific
to the true trajectory rather than generic -- algebraically the same quantity as the wrong-target
experiment, which was removed from the page. The row is correct but no longer has a section
explaining it.

The three PNGs are the across-scene vs within-scene length figures produced for review
(`plot_length_claim.py`, archived at `/mnt/efs/users/rod/results/llr_artifact/`); they are not
embedded in the page.
