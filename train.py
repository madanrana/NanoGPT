import torch
import json
import os
import config


def get_batch(data, block_size, batch_size, device):

    ix = torch.randint(len(data) - block_size, (batch_size,))

    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])

    return x.to(device), y.to(device)


def train_model(model, data, optimizer, device):

    model.train()

    loss_history = []

    for step in range(config.max_steps):

        xb, yb = get_batch(
            data,
            config.block_size,
            config.batch_size,
            device
        )

        logits, loss = model(xb, yb)

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        loss_value = loss.item()

        loss_history.append(loss_value)

        if step % config.log_interval == 0:

            print(f"Step {step} | Loss {loss_value:.4f}")

        # save checkpoint
        if step % config.checkpoint_interval == 0 and step != 0:

            checkpoint_path = f"checkpoints/model_step_{step}.pt"

            torch.save(model.state_dict(), checkpoint_path)

            print(f"Checkpoint saved: {checkpoint_path}")

    # save training log

    os.makedirs("logs", exist_ok=True)

    with open("logs/loss_history.json", "w") as f:

        json.dump(loss_history, f)

    print("Training logs saved.")