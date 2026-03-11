import torch
import config
import torch.nn.functional as F


def generate_text(model, start_token, decode, device, max_tokens=200):

    model.eval()

    context = torch.tensor([[start_token]], dtype=torch.long).to(device)

    for _ in range(max_tokens):

        context_cond = context[:, -config.block_size:]
        logits, _ = model(context_cond)

        logits = logits[:, -1, :]      # last token prediction
        
        logits = logits / config.temperature  # adjust randomness
        
        probs = torch.softmax(logits, dim=-1)
        
        next_token = torch.multinomial(probs, 1)

        context = torch.cat((context, next_token), dim=1)

    return decode(context[0].tolist())
