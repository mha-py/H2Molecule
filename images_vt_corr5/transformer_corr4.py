
from layers import *

Parameter:  n=128, nh=2

class Net(nn.Module):
    def __init__(self, n, nh, nout=7):
        super().__init__()

        self.nout = nout

        self.conv1 = nn.Conv2d(1, n//4, 5, padding=2, stride=2, padding_mode='reflect')
        #self.conv2 = nn.Conv2d(n//2, n, 3, padding=1, padding_mode='reflect')
        self.conv2 = nn.Conv2d(n//4, n//4, 3, padding=1, padding_mode='reflect')
        self.conv3 = nn.Conv2d(n//4, n//2, 3, padding=1, stride=2, padding_mode='reflect')
        self.predense = nn.Linear(4*4*n//2, n)

        self.posenc = PositionalEncoding2d(n)

        self.seed = Seed(n, nout)
        self.enc1 = EncoderBlock(n, n, nh)
        self.enc2 = EncoderBlock(n, n, nh)
        self.enc3 = EncoderBlock(n, n, nh)
        self.enc4 = EncoderBlock(n, n, nh)
        self.enc5 = EncoderBlock(n, n, nh)
        self.enc6 = EncoderBlock(n, n, nh)

        self.ln1 = LayerNorm(n)
        self.ln2 = LayerNorm(n)

        #self.dense = nn.Linear(n, nout)
        self.dense = nn.ModuleList([nn.Linear(n, 5) for i in range(nout)])

        self.dropout = nn.Dropout(0.1)
        self.cuda()

    def forward(self, x): # x = img, y = params
        x = rearrange(x, 'b h w c -> b c h w')
        x = relu(self.conv1(x))
        x = relu(self.conv2(x))
        x = relu(self.conv3(x))
        x = rearrange(x, 'b c h w -> b h w c')
        x = rearrange(x, 'b (h f1) (w f2) c -> b h w (f1 f2 c)', f1=4, f2=4)
        x = relu(self.predense(x))


        x = self.posenc(x)
        x = self.ln1(x)

        x = rearrange(x, 'b h w c -> b (h w) c')
        x = self.dropout(x)

        y = self.seed(x) # classifier token
        x = torch.cat((y, x), 1)
        x = self.enc1(x)
        x = self.enc2(x)
        x = self.enc3(x)
        #x = self.enc4(x)
        #x = self.enc5(x)
        #x = self.enc6(x)

        x = self.ln2(x)
        x = self.dropout(x)
        xs = [ self.dense[i](x[:,i,:]) for i in range(self.nout) ]
        z = torch.stack(xs, 1)

       # z = self.dense(x)
        return z
