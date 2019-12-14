import time
import torch
import torch.nn as nn
import torch.optim as optim
from models import tiramisu
import utils.training as train_utils
from datagenerator_new import CustomDataset
#gtFine_labelIds
path='./dataset/'
batch_size=1
N_EPOCHS=500
#phase='images/train'

LR = 1e-4
LR_DECAY = 0.995
DECAY_EVERY_N_EPOCHS = 1
torch.cuda.manual_seed(0)

def load_model(path=None):
	if path==None:
		model = tiramisu.FCDenseNet103(n_classes=30).cuda()
	model.apply(train_utils.weights_init)
	return model

print('loading model:')
model=load_model()
print('loading dataset:')
optimizer = torch.optim.RMSprop(model.parameters(), lr=LR, weight_decay=1e-4)
train_loader = torch.utils.data.DataLoader(CustomDataset(batch_size,path,'train'),batch_size, shuffle=True, num_workers=4)
print('number of iteration per epoch',len(train_loader))
#criterion = nn.NLLLoss2d().cuda()


for epoch in range(1, N_EPOCHS+1):
	since = time.time()

	### Train ###
	trn_loss = train_utils.train(model, train_loader, optimizer, epoch)
	print('Epoch {:d}\nTrain - Loss: {:.4f}'.format(epoch, trn_loss))
	time_elapsed = time.time() - since
	print('Train Time {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

	'''
	### Test ###
	val_loss, val_err = train_utils.test(model, val_loader, criterion, epoch)
	print('Val - Loss: {:.4f} | Acc: {:.4f}'.format(val_loss, 1-val_err))
	time_elapsed = time.time() - since
	print('Total Time {:.0f}m {:.0f}s\n'.format(time_elapsed // 60, time_elapsed % 60))
	'''
	### Checkpoint ###
	if epoch%50==0:
		train_utils.save_weights(model, epoch, trn_loss, trn_loss)

	### Adjust Lr ###
	train_utils.adjust_learning_rate(LR, LR_DECAY, optimizer, epoch, DECAY_EVERY_N_EPOCHS)
