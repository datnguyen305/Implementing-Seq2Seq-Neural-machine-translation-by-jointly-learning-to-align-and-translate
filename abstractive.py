import torch
from torch.nn import functional as F
import torch.nn as nn
import random
from vocabs.vocab import Vocab
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from builders.model_builder import META_ARCHITECTURE

class Encoder(nn.Module):
    def __init__(self, config, vocab: Vocab):
        super().__init__()

        self.embedding = nn.Embedding(vocab.vocab_size, config.hidden_size, padding_idx = vocab.vocab.pad_idx)
        self.rnn = nn.GRU(
            config.enc_emb_dim, 
            config.enc_hid_dim,
            bidirectional=True,
            batch_first=True
        )
        self.fc = nn.Linear(config.enc_hid_dim * 2, config.dec_hid_dim)
        self.dropout = nn.Dropout(config.Dropout)

    def forward(self, input):
        # input: [batch_size, src_len]
        input_len = input.shape[1]
        embedded = self.dropout(self.embedding(input))
        # embedded: [batch_size, src_len, emb_dim]
        packed_embedded = pack_padded_sequence(embedded, input_len.cpu() , batch_first=True, enforce_sorted=False)
        packed_outputs, hidden = self.rnn(packed_embedded)
        # hidden: [num_layers * num_directions, batch_size, dec_hid_dim]
        outputs,_ = pad_packed_sequence(packed_outputs, batch_first=True)
        # outputs: [batch_size, src_len, num_directions * enc_hid_dim]

        hidden = torch.tanh(self.fc(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)))
        # hidden: [batch_size, dec_hid_dim aka hidden_size*2]
        return outputs, hidden
    
class Attention(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.attn == nn.Linear((config.enc_hid_dim * 2) + config.dec_hid_dim, config.dec_hid_dim)
        self.v = nn.Linear(config.dec_hid_dim, 1, bias=False)

    def forward(self, hidden, encoder_outputs):
        # hidden: [batch_size, dec_hid_dim] 
        # encoder_outputs: [batch_size, src_len, enc_hid_dim * 2]

        src_len = encoder_outputs.shape[1]

        hidden = hidden.unsqueeze(1).repeat(1, src_len, 1)
        # hidden: [batch_size, src_len, dec_hid_dim]
        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2)))
        # energy: [batch_size, src_len, dec_hid_dim]  
        attention = self.v(energy).squeeze(2)
        # attention: [batch_size, src_len]
        return F.softmax(attention, dim=1)
    
class Decoder(nn.Module):
    def __init__(self, config, vocab: Vocab):
        super().__init__()
        
        self.output_dim = Vocab.vocab_size
        self.attention = Attention()
        
        self.embedding = nn.Embedding(self.output_dim, config.dec_emb_dim, padding_idx=vocab.pad_idx)
        self.rnn = nn.GRU((config.enc_hid_dim * 2) + config.dec_emb_dim, config.dec_hid_dim, batch_first=True)
        self.fc_out = nn.Linear((config.enc_hid_dim * 2) + config.dec_hid_dim + config.dec_emb_dim, self.output_dim)
        self.dropout = nn.Dropout(config.dropout)
    
    def forward(self, input, hidden , encoder_outputs):
        # input: [batch_size]: y_i-1
        # hidden: [batch_size, dec_hid_dim]: s_i-1
        # encoder_outputs: [batch_size, src_len, enc_hid_dim * 2]: h_j

        input = input.unsqueeze(1)
        # input: [batch_size, 1]
        embedded = self.dropout(self.embedding(input))
        # embedded: [batch_size, 1, dec_emb_dim]
        a = self.attention(hidden, encoder_outputs)
        # a [batch_size, src_len]
        a = a.unsqueeze(1)
        # a [batch_size, 1, src_len]
        # encoder_outputs: [batch_size, src_len, enc_hid_dim * 2]
        weight = a.bmm(encoder_outputs)
        # weight: [batch_size, 1, enc_hid_dim * 2]
        rnn_input = torch.cat((embedded,weight), dim=2)
        # rnn_input: [batch_size, 1, dec_emb_dim + enc_hid_dim * 2]
        # hidden: [batch_size, dec_hid_dim]
        output, hidden = self.rnn(rnn_input, hidden.unsqueeze(0))
        # output: [batch_size, 1, dec_hid_dim]
        # hidden: [1, batch_size, dec_hid_dim]
        # weights: [batch_size, 1, enc_hid_dim * 2]
        prediction = self.fc_out(torch.cat((output, weight, embedded), dim=2))
        # prediction: [batch_size, 1, output_dim]
        prediction = prediction.squeeze(1)

        return prediction, hidden.squeeze(0), a.squeeze(1)

class Seq2Seq(nn.Module):
    def __init__(self, config, vocab):
        super().__init__()
        
        self.encoder = Encoder()
        self.decoder = Decoder()
        
        self.MAX_LENGTH = vocab.max_sentence_length + 2 # + 2 for bos and eos tokens

        self.device = config.device
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=vocab.pad_idx)
        
    def forward(self, src, trg): # teacher forcing = 0.5
        # src = [batch size, src len]
        # trg = [batch size, trg len]
        teacher_forcing_ratio = 0.5
        batch_size = src.shape[0]
        trg_len = trg.shape[1]
        trg_vocab_size = self.vocab.vocab_size
        
        outputs = torch.zeros(batch_size, trg_len, trg_vocab_size).to(self.device)
        
        encoder_outputs, hidden = self.encoder(src)
        
        input = trg[:, 0]  # First input is <sos> token
        for t in range(self.MAX_LENGTH):
            output, hidden, _ = self.decoder(input, hidden, encoder_outputs)
            outputs[:, t] = output
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = output.argmax(1)
            input = trg[:, t] if teacher_force else top1
            
        loss = self.loss_fn(outputs[:, 1:].reshape(-1, trg_vocab_size), trg[:, 1:].reshape(-1))
        
        return outputs, loss
    
    def predict(self, src, max_len=50):
        """Generate translation without teacher forcing"""
        with torch.no_grad():
            batch_size = src.shape[0]
            src_len = src.shape[1]
            
            encoder_outputs, hidden = self.encoder(src, src_len)
            
            outputs = torch.zeros(batch_size, max_len, self.vocab.vocab_size).to(self.device)
            input = torch.tensor([self.vocab.bos_idx] * batch_size).to(self.device)
            
            for t in range(self.MAX_LENGTH):
                output, hidden, _ = self.decoder(input, hidden, encoder_outputs)
                outputs[:, t] = output
                input = output.argmax(1)
                
                # Stop if all sequences generated <eos>
                if (input == self.vocab.eos_idx).all():
                    break
            
            return outputs.argmax(2)  # [batch_size, max_len]
