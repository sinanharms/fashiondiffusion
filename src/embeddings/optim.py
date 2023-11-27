from typing import List

import torch
from tqdm import tqdm


def optimize_embeddings(
    model,
    opt: torch.optim,
    loss: torch.nn.functional,
    emb: List,
    iterations: int,
    device,
    init_latent,
):
    for i in tqdm(range(iterations)):
        opt.zero_grad()

        noise = torch.randint_like(init_latent)
        target_enc = torch.randint(1000, (1,), device=device)
        z = model.q_sample(init_latent, target_enc, noise=noise)

        pred_noise = model.apply_model(z, target_enc, emb)

        loss = loss(pred_noise, noise)
        loss.backward()
        tqdm.set_postfix({"loss": loss.item()})
        history.append(loss.item())
        opt.step()
