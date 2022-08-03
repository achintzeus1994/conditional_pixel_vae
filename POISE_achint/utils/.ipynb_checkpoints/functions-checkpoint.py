import torch
import matplotlib.pyplot as plt

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def train(model, joint_dataloader, optimizer, epoch, generate_mode):
    model.train()
    for k, data in enumerate(joint_dataloader):
        optimizer.zero_grad()
        results = model([data[i].to(device=device) for i in range(model.M)], generate_mode)
        results['total_loss'].backward() 
        optimizer.step()
        

def test(model, joint_dataloader, optimizer, epoch, generate_mode):
    model.eval()
    with torch.no_grad():
        for k, data in enumerate(joint_dataloader):
            optimizer.zero_grad()
            results = model([data[i].to(device=device) for i in range(model.M)], generate_mode)
            

def plotting(model, data, x_rec):
    ncols = min(len(x_rec[0]), 20)
    fig, ax = plt.subplots(nrows=2*model.M, ncols=ncols, figsize=(15, 3))
    for m in range(model.M): 
        for i, aux in enumerate(zip(data[m], x_rec[m])):
            if i >= ncols:
                break
            for j, im in enumerate(aux):
                ax[j+2*m, i].imshow(im.cpu().numpy().reshape(28, 28), cmap='gray')
                ax[j+2*m, i].set_axis_off()
    fig.tight_layout(pad=0)