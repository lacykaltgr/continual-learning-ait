import classifier_lib
import torch
import numpy as np
import torchvision.transforms as transforms
import torch.utils.data as data


class RealFakeConditionalDataset(data.Dataset):
  def __init__(self, x_np, y_np, cond_np, transform=transforms.ToTensor()):
    super(RealFakeConditionalDataset, self).__init__()

    self.x = x_np
    self.y = y_np
    self.cond = cond_np
    self.transform = transform

  def __getitem__(self, index):
    return self.transform(self.x[index]), self.y[index], self.cond[index]

  def __len__(self):
    return len(self.x)

def train_generator(
        X_real, y_real,
        X_gen, y_gen,
        encoder, discriminator,
        epochs=10, batch_size=256,
        learning_rate=1e-4,
        device='cuda:0',
):
    # Combine the fake / real
    train_data = np.concatenate((X_real, X_gen))
    train_label = torch.zeros(train_data.shape[0])
    train_label[:X_real.shape[0]] = 1.
    transform = transforms.Compose([transforms.ToTensor()])

    condition_label = np.concatenate((y_real, y_gen))
    train_dataset = RealFakeConditionalDataset(train_data, train_label, condition_label, transform)

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

    # Prepare training
    vpsde = classifier_lib.vpsde()
    optimizer = torch.optim.Adam(discriminator.parameters(), lr=learning_rate, weight_decay=1e-7)
    loss = torch.nn.BCELoss()
    scaler = lambda x: 2. * x - 1.

    # Training
    for i in range(epochs):
        outs = []
        cors = []
        for data in train_loader:
            optimizer.zero_grad()

            inputs, labels, cond = data
            cond = cond.to(device)
            inputs = inputs.to(device)
            labels = labels.to(device)
            inputs = scaler(inputs)

            # Data perturbation
            t, _ = vpsde.get_diffusion_time(inputs.shape[0], inputs.device)
            mean, std = vpsde.marginal_prob(t)
            z = torch.randn_like(inputs)
            perturbed_inputs = mean[:, None, None, None] * inputs + std[:, None, None, None] * z

            # Forward
            with torch.no_grad():
                pretrained_feature = encoder(perturbed_inputs, timesteps=t, feature=True)
            label_prediction = discriminator(pretrained_feature, t, sigmoid=True, condition=cond).view(-1)

            # Backward
            out = loss(label_prediction, labels)
            out.backward()
            optimizer.step()

            # Report
            cor = ((label_prediction > 0.5).float() == labels).float().mean()
            outs.append(out.item())
            cors.append(cor.item())
            print(f"{i}-th epoch BCE loss: {np.mean(outs)}, correction rate: {np.mean(cors)}")