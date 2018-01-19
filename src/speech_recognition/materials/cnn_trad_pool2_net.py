from common import*

# Cnn_Trad_Pool2_Net(
#   (conv1): Conv2d (1, 64, kernel_size=[20, 8], stride=(1, 1))
#   (pool1): MaxPool2d(kernel_size=(2, 2), stride=(2, 2), dilation=(1, 1))
#   (conv2): Conv2d (64, 64, kernel_size=[10, 4], stride=(1, 1))
#   (pool2): MaxPool2d(kernel_size=(1, 1), stride=(1, 1), dilation=(1, 1))
#   (output): Linear(in_features=26624, out_features=12)
#   (dropout): Dropout(p=0.5)
# )


# load from marvis pretrained model
def load_pretrain_file(net, pretrain_file, skip=[]):

    pretrain_state_dict = torch.load(pretrain_file)
    state_dict = net.state_dict()
    keys = list(state_dict.keys())
    for key in keys:
        if key in skip:
            continue

        pretrain_key = key
        if 'fc.' in key: pretrain_key = key.replace('fc.','output.')

        if pretrain_key is not None:
            #print('%36s,%36s'%(key,pretrain_key))
            #print('%36s,%36s,  %s,%s'%(key,pretrain_key,str(state_dict[key].size()),str(pretrain_state_dict[pretrain_key].size())))
            state_dict[key] = pretrain_state_dict[pretrain_key]

    net.load_state_dict(state_dict)



class Cnn_Trad_Pool2_Net(nn.Module):
    def __init__(self, in_shape=(1,40,101), num_classes=12 ):
        super(Cnn_Trad_Pool2_Net, self).__init__()
        self.num_classes = num_classes

        self.conv1 = nn.Conv2d(1,  64, kernel_size=(20, 8), stride=(1, 1))
        self.conv2 = nn.Conv2d(64, 64, kernel_size=(10, 4), stride=(1, 1))
        self.fc = nn.Linear(26624,num_classes)


    def forward(self, x):

        x = self.conv1(x)
        x = F.relu(x,inplace=True)
        x = F.max_pool2d(x,kernel_size=(2,2),stride=(2,2))

        x = self.conv2(x)
        x = F.relu(x,inplace=True)
        x = x.view(x.size(0), -1)

        x = F.dropout(x,p=0.5,training=self.training)
        x = self.fc(x)

        return x  #logits



def run_check_net():

    # https://discuss.pytorch.org/t/print-autograd-graph/692/8
    batch_size  = 32
    num_classes = 12
    H = 40
    W = 101
    labels = torch.randn(batch_size,num_classes)
    inputs = torch.randn(batch_size,1,H,W)
    y = Variable(labels).cuda()
    x = Variable(inputs).cuda()


    net = Cnn_Trad_Pool2_Net(in_shape=(1,H,W), num_classes=num_classes)
    net.cuda()
    net.train()


    logits = net.forward(x)
    probs  = F.softmax(logits, dim=1)

    loss = F.binary_cross_entropy_with_logits(logits, y)
    loss.backward()

    print(type(net))
    #print(net)
    print('probs')
    print(probs)

def run_check_pretrain():

    batch_size  = 32
    num_classes = 12
    H = 40
    W = 101
    net = Cnn_Trad_Pool2_Net(in_shape=(1,H,W), num_classes=num_classes)

    pretrain_file='/root/share/project/kaggle/tensorflow/build/__reference__/honk1/model/google-speech-dataset.pt'
    load_pretrain_file(net, pretrain_file)

########################################################################################
if __name__ == '__main__':
    print( '%s: calling main function ... ' % os.path.basename(__file__))

    #run_check_net()
    run_check_pretrain()

