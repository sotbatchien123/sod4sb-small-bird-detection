"""Retry an accumulated AMP batch without dropping images or updating bad weights."""
import torch


def recover_gradients(model, optimizer, scaler, batches, replay, buffers, rng,
                      on_retry, max_retries=8):
    end_rng=(torch.get_rng_state(),torch.cuda.get_rng_state())
    for attempt in range(max_retries + 1):
        scaler.unscale_(optimizer)
        finite = torch.stack([torch.isfinite(p.grad).all() for p in model.parameters()
                              if p.grad is not None]).all().item()
        if finite:
            if attempt:
                torch.set_rng_state(end_rng[0])
                torch.cuda.set_rng_state(end_rng[1])
            return torch.nn.utils.clip_grad_norm_(model.parameters(), 35., error_if_nonfinite=True)
        if attempt == max_retries:
            raise RuntimeError('Non-finite gradients persist after 8 AMP retries')
        old_scale = scaler.get_scale()
        # unscale_ recorded inf/nan, so step deliberately skips the optimizer.
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        with torch.no_grad():
            for name, value in model.named_buffers():
                value.copy_(buffers[name])
        torch.set_rng_state(rng[0])
        torch.cuda.set_rng_state(rng[1])
        on_retry(attempt + 1, old_scale, scaler.get_scale())
        for batch in batches:
            replay(batch)
