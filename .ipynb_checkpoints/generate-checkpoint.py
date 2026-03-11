import torch
import torch.nn.functional as F


def generate_text(model, start_token, decode, device, max_tokens=200):

    model.eval()

    context = torch.tensor([[start_token]], dtype=torch.long).to(device)

    for _ in range(max_tokens):

        logits, _ = model(context)

        probs = F.softmax(logits[:, -1, :], dim=-1)

        next_token = torch.multinomial(probs, 1)

        context = torch.cat((context, next_token), dim=1)

    return decode(context[0].tolist())
