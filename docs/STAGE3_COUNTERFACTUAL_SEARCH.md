# Stage 3 v5: Counterfactual Market Search

This stage replaces PPO after repeated evaluations showed that its deterministic
policy remained identical to the incumbent. It does not optimize a neural loss.
Instead, it compares concrete market rules through paired games with identical
seeds, opponents, and seats.

Each candidate can only add a premium-product sale when the incumbent was not
already selling that product. Its decision uses current stock, price relative to
the base price, town demand, recent price movement, and the in-game day. Routes,
workers, production, purchases, and every existing incumbent sale stay unchanged.

The search has two independent parts:

1. Twelve candidates play a small screening set against the incumbent's two
   weakest league opponents.
2. The best two candidates play all six opponents on separate held-out seeds and
   both seats.

Promotion still requires zero runtime errors, a changed market action, an overall
score/margin improvement, and no material collapse against veto opponents. If no
candidate passes, the packaged policy is an exact incumbent fallback.

Open `kaggriculture_stage3_counterfactual_colab.ipynb`. Results and the streamed
log are written to:

`MyDrive/Kaggriculture/results/stage3_counterfactual_market_v5/`

The method is CPU-oriented and does not require a GPU. Kaggle submission remains
disabled by default.
