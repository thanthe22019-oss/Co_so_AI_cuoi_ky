from torch import no_grad
from torch.utils.data import DataLoader


"""
Functions you should use.
Please avoid importing any other functions or modules.
Your code will not pass if the gradescope autograder detects any changed imports
"""
from torch import optim, tensor
from losses import regression_loss, digitclassifier_loss, languageid_loss, digitconvolution_Loss
from torch import movedim


"""
##################
### QUESTION 1 ###
##################
"""


def train_perceptron(model, dataset):
    """
    Train the perceptron until convergence.
    You can iterate through DataLoader in order to 
    retrieve all the batches you need to train on.

    Each sample in the dataloader is in the form {'x': features, 'label': label} where label
    is the item we need to predict based off of its features.
    """
    with no_grad():
        dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
        "*** YOUR CODE HERE ***"
        while True:
            mistakes = 0
            for batch in dataloader:
                x = batch['x'].squeeze(0)
                label = batch['label'].item()
                if model.get_prediction(x) != label:
                    model.w.add_(label * x)
                    mistakes += 1
            if mistakes == 0:
                break


def train_regression(model, dataset):
    """
    Trains the model.

    In order to create batches, create a DataLoader object and pass in `dataset` as well as your required 
    batch size. You can look at PerceptronModel as a guideline for how you should implement the DataLoader

    Each sample in the dataloader object will be in the form {'x': features, 'label': label} where label
    is the item we need to predict based off of its features.

    Inputs:
        model: Pytorch model to use
        dataset: a PyTorch dataset object containing data to be trained on
        
    """
    "*** YOUR CODE HERE ***"
    optimizer = optim.Adam(model.parameters(), lr=.003)
    loader = DataLoader(dataset, batch_size=200, shuffle=True)
    for epoch in range(3000):
        for batch in loader:
            optimizer.zero_grad()
            loss = regression_loss(model(batch['x']), batch['label'])
            loss.backward()
            optimizer.step()
        if loss.item() < .008:
            break
    print('Regression epochs:', epoch + 1, 'loss:', loss.item())


def train_digitclassifier(model, dataset):
    """
    Trains the model.
    """
    model.train()
    """ YOUR CODE HERE """
    optimizer = optim.Adam(model.parameters(), lr=.001)
    loader = DataLoader(dataset, batch_size=128, shuffle=True)
    for epoch in range(12):
        for batch in loader:
            optimizer.zero_grad()
            loss = digitclassifier_loss(model(batch['x']), batch['label'])
            loss.backward()
            optimizer.step()
        with no_grad():
            accuracy = dataset.get_validation_accuracy()
        print('Digits epoch:', epoch + 1, 'dev accuracy:', accuracy)
        if accuracy >= .98:
            break


def train_languageid(model, dataset):
    """
    Trains the model.

    Note that when you iterate through dataloader, each batch will returned as its own vector in the form
    (batch_size x length of word x self.num_chars). However, in order to run multiple samples at the same time,
    get_loss() and run() expect each batch to be in the form (length of word x batch_size x self.num_chars), meaning
    that you need to switch the first two dimensions of every batch. This can be done with the movedim() function 
    as follows:

    movedim(input_vector, initial_dimension_position, final_dimension_position)

    For more information, look at the pytorch documentation of torch.movedim()
    """
    model.train()
    "*** YOUR CODE HERE ***"
    optimizer = optim.Adam(model.parameters(), lr=.001)
    # Use length buckets: no padding overhead and no test-set selection.
    for epoch in range(30):
        for bucket_start, bucket_end in dataset.train_buckets:
            indices = DataLoader(range(int(bucket_start), int(bucket_end)), batch_size=128, shuffle=True)
            for index in indices:
                xs, labels = dataset._encode(dataset.train_x[index.tolist()], dataset.train_y[index.tolist()])
                optimizer.zero_grad()
                loss = languageid_loss(model(xs), labels)
                loss.backward()
                optimizer.step()
        with no_grad():
            correct = 0
            for start, end in dataset.dev_buckets:
                xs, labels = dataset._encode(dataset.dev_x[start:end], dataset.dev_y[start:end])
                correct += (model(xs).argmax(dim=1) == labels.argmax(dim=1)).sum().item()
            accuracy = correct / len(dataset.dev_y)
        print('Language epoch:', epoch + 1, 'dev accuracy:', accuracy)
        if accuracy >= .86:
            break



def Train_DigitConvolution(model, dataset):
    """
    Trains the model.
    """
    """ YOUR CODE HERE """
    optimizer = optim.Adam(model.parameters(), lr=.003)
    loader = DataLoader(dataset, batch_size=50, shuffle=True)
    for _ in range(35):
        for batch in loader:
            optimizer.zero_grad()
            loss = digitconvolution_Loss(model(batch['x']), batch['label'])
            loss.backward()
            optimizer.step()
