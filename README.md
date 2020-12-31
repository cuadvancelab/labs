ADVANCE LABs

Lab1 -- text-based cyberharassment detection: 

We illustrate how to set up a laboratory for the text-based cyberharassment detection lab. We will first give an introduction of the interface for the AI model architectures, then we will provide the training algorithm description and implementation details of the training algorithm. In ADVANCE labs, all the default parameters are pre-defined and saved in configs. The get_data_loaders() function provides an easy access to public benchmark datasets such as cyberbullying, cyberharassment and hate speech datasets, including our datasets of cyberbullying images and boomer-hate speech. The get_model(), get_criterion() and get_optimizer() function provide convenient access to the AI model, the loss function and the optimization algorithm used in training the AI model. Next, the AI model is trained in the training loop (model.train) and evaluated in the testing loop (model.eval). This lab covers three representative training architectures including CNN, RNN and BERT, all of which can be easily used by passing relevant parameters in the helper functions mentioned above. The full source code of ADVANCE will be open-sourced on Github, which would make further development flexible and extendable.


from lab_utils.data.helpers import *

train_loader, val_loader, test_loader = get_data_loaders()
model = get_model()
criterion = get_criterion()
optimizer = get_optimizer(model)

for i_epoch in range(0, 100):
    model.train()
    optimizer.zero_grad()

    for batch in tqdm(train_loader, total = len(train_loader)):
        loss = model_forward(i_epoch, model, criterion, batch)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

model.eval()
model_eval(test_loader, model, criterion)



