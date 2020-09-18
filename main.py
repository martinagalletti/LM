from batches import *
from loss_function import *
from model import *
from preprocess import *
from prediction import *
from argparse import Namespace

flags = Namespace(
    train_file='final_df.txt',
    seq_size=32,
    batch_size=16,
    embedding_size=64,
    lstm_size=64,
    gradients_norm=5,
    # initial_words=['I', 'am'],
    predict_top_k=5,
    checkpoint_path='checkpoint',
)


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # transfer to GPU if there
    int_to_vocab, vocab_to_int, n_vocab, in_text, out_text = get_data_from_file(
        flags.train_file, flags.batch_size, flags.seq_size)  # get variables

    net = RNNModule(n_vocab, flags.seq_size,
                    flags.embedding_size, flags.lstm_size)  # call the model
    net = net.to(device)

    criterion, optimizer = get_loss_and_train_op(net, 0.01)  # loss function

    iteration = 0

    # for each epoch, loop through the batches to compute loss values
    # + update network’s parameters.

    # Call the train() method on the network’s instance
    # (it will inform inner mechanism that we are about to train, not execute the training)
    # Reset all gradients, Compute output, loss value, accuracy, etc
    # Perform back-propagation,
    # Update the network’s parameters

    for e in range(50):
        batches = get_batches(in_text, out_text, flags.batch_size, flags.seq_size)
        state_h, state_c = net.zero_state(flags.batch_size)

        # Transfer data to GPU
        state_h = state_h.to(device)
        state_c = state_c.to(device)
        for x, y in batches:
            iteration += 1

            # Tell it we are in training mode
            net.train()

            # Reset all gradients
            optimizer.zero_grad()

            # Transfer data to GPU
            x = torch.tensor(x).to(device)
            y = torch.tensor(y).to(device)

            logits, (state_h, state_c) = net(x, (state_h, state_c))
            loss = criterion(logits.transpose(1, 2), y)

            state_h = state_h.detach()
            state_c = state_c.detach()

            loss_value = loss.item()

            # You may notice the detach() thing. Whenever we want to use something that belongs to the computational graph
            # for other operations, we must remove them from the graph by calling detach() method.
            # The reason is, Pytorch keeps track of the tensors’ flow to perform back-propagation through a mechanism
            # called autograd. We mess it up and Pytorch will fail to deliver the loss.

            # Perform back-propagation
            loss.backward()

            _ = torch.nn.utils.clip_grad_norm_(
                net.parameters(), flags.gradients_norm)

            # Update the network's parameters
            optimizer.step()

        if iteration % 100 == 0:
            print('Epoch: {}/{}'.format(e, 200),
                  'Iteration: {}'.format(iteration),
                  'Loss: {}'.format(loss_value))

        if iteration % 1000 == 0:
            predict(device, net, flags.initial_words, n_vocab,
                    vocab_to_int, int_to_vocab, top_k=5)
            torch.save(net.state_dict(),
                       'checkpoint_pt/model-{}.pth'.format(iteration))