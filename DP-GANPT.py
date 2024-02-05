#!/usr/bin/env python
# coding: utf-8

# Required Imports.




import torch
import io
import torch.nn.functional as F
import random
import numpy as np
import time
import math
import datetime
import matplotlib.pyplot as plt
import torch.nn as nn
from transformers import *
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler



seed_val = 42
random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)
if torch.cuda.is_available():
  torch.cuda.manual_seed_all(seed_val)




if torch.cuda.is_available():    

    device = torch.device("cuda")
    print('There are %d GPU(s) available.' % torch.cuda.device_count())
    print('We will use the GPU:', torch.cuda.get_device_name(0))
# If not...
else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")






#--------------------------------
#  Transformer parameters
#--------------------------------
max_seq_length = 512
batch_size = 16



num_hidden_layers_g = 1; 

num_hidden_layers_d = 1; 

noise_size = 100

out_dropout_rate = 0.3


apply_balance = False

#--------------------------------
#  Optimization parameters
#--------------------------------
learning_rate_discriminator = 2e-6
learning_rate_generator = 2e-6
epsilon = 1e-8
num_train_epochs = 20
multi_gpu = True
# Scheduler
apply_scheduler = False
warmup_proportion = 0.1
# Print
print_each_n_step = 10



model_name = "microsoft/codebert-base"

#  NOTE: in this setting 50 classes are involved
labeled_file = "E:\\dataset\\camel-train.txt"

unlabeled_file0 = "E:\\dataset\\jedit-train.txt"
unlabeled_file1 = "E:\\dataset\\lucene-train.txt"
unlabeled_file2 = "E:\\dataset\\xalan-train.txt"


test_filename = "E:\\dataset\\camel-test.txt"

label_list = ["0","1"]


# Load the Tranformer Model

# In[4]:


transformer = AutoModel.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)


# Function required to load the dataset

# In[5]:


def get_sdp_examples(input_file):
  """Creates examples for the training and dev sets."""
  examples = []

  with open(input_file, 'r') as f:
      contents = f.read()
      file_as_list = contents.splitlines()
      for line in file_as_list[1:]:
          split = line.split("<CODESPLIT>")
          code = split[4]
          nl = ' '.join(split[1:4])
          label = split[0]
          examples.append((nl, code, label))
      f.close()

  return examples


# **Load** the input dataset


#Load the examples
labeled_examples = get_sdp_examples(labeled_file)
unlabeled_examples = get_sdp_examples(unlabeled_file0) + get_sdp_examples(unlabeled_file1)+ get_sdp_examples(unlabeled_file2)
test_examples = get_sdp_examples(test_filename)
random.shuffle(labeled_examples)
random.shuffle(unlabeled_examples)
random.shuffle(test_examples)




def generate_data_loader(input_examples, label_masks, label_map, do_shuffle = False, balance_label_examples = False):
  '''
  Generate a Dataloader given the input examples, eventually masked if they are 
  to be considered NOT labeled.
  '''
  examples = []

  num_labeled_examples = 0
  for label_mask in label_masks:
    if label_mask: 
      num_labeled_examples += 1
  label_mask_rate = num_labeled_examples/len(input_examples)


  for index, ex in enumerate(input_examples): 

    if label_mask_rate == 1 or not balance_label_examples:
      examples.append((ex, label_masks[index]))
    else:

      if label_masks[index]:
        balance = int(1/label_mask_rate)
        balance = int(math.log(balance,2))
        if balance < 1:
          balance = 1
        for b in range(0, int(balance)):
          examples.append((ex, label_masks[index]))
      else:
        examples.append((ex, label_masks[index]))
  

  input_ids = []
  input_mask_array = []
  label_mask_array = []
  label_id_array = []

  # Tokenization 
  for (text, label_mask) in examples:
    encoded_sent = tokenizer.encode(text = text[0], text_pair = text[1], add_special_tokens=True, max_length=max_seq_length, padding="max_length", truncation=True)
    input_ids.append(encoded_sent)
    label_id_array.append(label_map[text[2]])
    label_mask_array.append(label_mask)
  
  # Attention to token (to ignore padded input wordpieces)
  for sent in input_ids:
    att_mask = [int(token_id > 0) for token_id in sent]                          
    input_mask_array.append(att_mask)
  # Convertion to Tensor
  input_ids = torch.tensor(input_ids) 
  input_mask_array = torch.tensor(input_mask_array)
  label_id_array = torch.tensor(label_id_array, dtype=torch.long)
  label_mask_array = torch.tensor(label_mask_array)

  # Building the TensorDataset
  dataset = TensorDataset(input_ids, input_mask_array, label_id_array, label_mask_array)

  if do_shuffle:
    sampler = RandomSampler
  else:
    sampler = SequentialSampler

  # Building the DataLoader
  return DataLoader(
              dataset,  # The training samples.
              sampler = sampler(dataset), 
              batch_size = batch_size) # Trains with this batch size.

def format_time(elapsed):
    '''
    Takes a time in seconds and returns a string hh:mm:ss
    '''
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))
    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))




label_map = {}
for (i, label) in enumerate(label_list):
  label_map[label] = i

train_examples = labeled_examples
train_label_masks = np.ones(len(labeled_examples), dtype=bool)


if unlabeled_examples:
  train_examples = train_examples + unlabeled_examples

  tmp_masks = np.zeros(len(unlabeled_examples), dtype=bool)
  train_label_masks = np.concatenate([train_label_masks,tmp_masks])

train_dataloader = generate_data_loader(train_examples, train_label_masks, label_map, do_shuffle = True, balance_label_examples = apply_balance)


test_label_masks = np.ones(len(test_examples), dtype=bool)

test_dataloader = generate_data_loader(test_examples, test_label_masks, label_map, do_shuffle = False, balance_label_examples = False)


# define the Generator and Discriminator 



#------------------------------
#   The Generator 
#------------------------------
class Generator(nn.Module):
    def __init__(self, noise_size=100, output_size=512, hidden_sizes=[512], dropout_rate=0.1):
        super(Generator, self).__init__()
        layers = []
        hidden_sizes = [noise_size] + hidden_sizes
        for i in range(len(hidden_sizes)-1):
            layers.extend([nn.Linear(hidden_sizes[i], hidden_sizes[i+1]), nn.LeakyReLU(0.2, inplace=True), nn.Dropout(dropout_rate)])

        layers.append(nn.Linear(hidden_sizes[-1],output_size))
        self.layers = nn.Sequential(*layers)

    def forward(self, noise):
        output_rep = self.layers(noise)
        return output_rep

#------------------------------
#   The Discriminator
#------------------------------
class Discriminator(nn.Module):
    def __init__(self, input_size=512, hidden_sizes=[512], num_labels=2, dropout_rate=0.1):
        super(Discriminator, self).__init__()
        self.input_dropout = nn.Dropout(p=dropout_rate)
        layers = []
        hidden_sizes = [input_size] + hidden_sizes
        for i in range(len(hidden_sizes)-1):
            layers.extend([nn.Linear(hidden_sizes[i], hidden_sizes[i+1]), nn.LeakyReLU(0.2, inplace=True), nn.Dropout(dropout_rate)])

        self.layers = nn.Sequential(*layers) #per il flatten
        self.logit = nn.Linear(hidden_sizes[-1],num_labels+1) # +1 for the probability of this sample being fake/real.
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, input_rep):
        input_rep = self.input_dropout(input_rep)
        last_rep = self.layers(input_rep)
        logits = self.logit(last_rep)
        probs = self.softmax(logits)
        return last_rep, logits, probs


# We instantiate the Discriminator and Generator

config = AutoConfig.from_pretrained(model_name)
hidden_size = int(config.hidden_size)

hidden_levels_g = [hidden_size for i in range(0, num_hidden_layers_g)]
hidden_levels_d = [hidden_size for i in range(0, num_hidden_layers_d)]

#-------------------------------------------------
#   Instantiate the Generator and Discriminator
#-------------------------------------------------
generator = Generator(noise_size=noise_size, output_size=hidden_size, hidden_sizes=hidden_levels_g, dropout_rate=out_dropout_rate)
discriminator = Discriminator(input_size=hidden_size, hidden_sizes=hidden_levels_d, num_labels=len(label_list), dropout_rate=out_dropout_rate)

# Put everything in the GPU if available
if torch.cuda.is_available():    
  generator.cuda()
  discriminator.cuda()
  transformer.cuda()
  if multi_gpu:
    transformer = torch.nn.DataParallel(transformer)

# print(config)





training_stats = []
training_st1 = []
exa = 0
best_f1 = 0
es = 0
total_t0 = time.time()

#models parameters
transformer_vars = [i for i in transformer.parameters()]
d_vars = transformer_vars + [v for v in discriminator.parameters()]
g_vars = [v for v in generator.parameters()]

#optimizer
dis_optimizer = torch.optim.AdamW(d_vars, lr=learning_rate_discriminator)
gen_optimizer = torch.optim.AdamW(g_vars, lr=learning_rate_generator) 

#scheduler
if apply_scheduler:
  num_train_examples = len(train_examples)
  num_train_steps = int(num_train_examples / batch_size * num_train_epochs)
  num_warmup_steps = int(num_train_steps * warmup_proportion)

  scheduler_d = get_cosine_with_hard_restarts_schedule_with_warmup(dis_optimizer, 
                                                               num_warmup_steps = num_warmup_steps,
                                                               num_training_steps = num_train_steps,
                                                                  num_cycle = 2)
  scheduler_g = get_cosine_with_hard_restarts_schedule_with_warmup(gen_optimizer, 
                                                               num_warmup_steps = num_warmup_steps,
                                                               num_training_steps = num_train_steps,
                                                                  num_cycle = 2)

# For each epoch...
for epoch_i in range(0, num_train_epochs):
    # ========================================
    #               Training
    # ========================================
    # Perform one full pass over the training set.
    print("")
    print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, num_train_epochs))
    print('Training...')

    # Measure how long the training epoch takes.
    t0 = time.time()

    # Reset the total loss for this epoch.
    tr_g_loss = 0
    tr_d_loss = 0

    # Put the model into training mode.
    transformer.train() 
    generator.train()
    discriminator.train()

    # For each batch of training data...
    for step, batch in enumerate(train_dataloader):

        exa = exa + 1
        # Progress update every print_each_n_step batches.
        if step % print_each_n_step == 0 and not step == 0:
            # Calculate elapsed time in minutes.
            elapsed = format_time(time.time() - t0)
            
            # Report progress.
            print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_dataloader), elapsed))

        # Unpack this training batch from our dataloader. 
        b_input_ids = batch[0].to(device)
        b_input_mask = batch[1].to(device)
        b_labels = batch[2].to(device)
        b_label_mask = batch[3].to(device)

        real_batch_size = b_input_ids.shape[0]
     
        # Encode real data in the Transformer
        model_outputs = transformer(b_input_ids, attention_mask=b_input_mask)
        hidden_states = model_outputs[-1]
        
        # Generate fake data that should have the same distribution of the ones
        # encoded by the transformer. 
        # First noisy input are used in input to the Generator
        noise = torch.zeros(real_batch_size, noise_size, device=device).uniform_(0, 1)
        # Gnerate Fake data
        gen_rep = generator(noise)

        # Generate the output of the Discriminator for real and fake data.
        # First, we put together the output of the tranformer and the generator
        disciminator_input = torch.cat([hidden_states, gen_rep], dim=0)
        # Then, we select the output of the disciminator
        features, logits, probs = discriminator(disciminator_input)

        # Finally, we separate the discriminator's output for the real and fake
        # data
        features_list = torch.split(features, real_batch_size)
        D_real_features = features_list[0]
        D_fake_features = features_list[1]
      
        logits_list = torch.split(logits, real_batch_size)
        D_real_logits = logits_list[0]
        D_fake_logits = logits_list[1]
        
        probs_list = torch.split(probs, real_batch_size)
        D_real_probs = probs_list[0]
        D_fake_probs = probs_list[1]

        #---------------------------------
        #  LOSS evaluation
        #---------------------------------
        # Generator's LOSS estimation
        g_loss_d = -1 * torch.mean(torch.log(1 - D_fake_probs[:,-1] + epsilon))
        g_feat_reg = torch.mean(torch.pow(torch.mean(D_real_features, dim=0) - torch.mean(D_fake_features, dim=0), 2))
        g_loss = g_loss_d + g_feat_reg
  
        # Disciminator's LOSS estimation
        logits = D_real_logits[:,0:-1]
        log_probs = F.log_softmax(logits, dim=-1)
        # The discriminator provides an output for labeled and unlabeled real data
        # so the loss evaluated for unlabeled data is ignored (masked)
        label2one_hot = torch.nn.functional.one_hot(b_labels, len(label_list))
        per_example_loss = -torch.sum(label2one_hot * log_probs, dim=-1)
        per_example_loss = torch.masked_select(per_example_loss, b_label_mask.to(device))
        labeled_example_count = per_example_loss.type(torch.float32).numel()

        # It may be the case that a batch does not contain labeled examples, 
        # so the "supervised loss" in this case is not evaluated
        if labeled_example_count == 0:
          D_L_Supervised = 0
        else:
          D_L_Supervised = torch.div(torch.sum(per_example_loss.to(device)), labeled_example_count)
                 
        D_L_unsupervised1U = -1 * torch.mean(torch.log(1 - D_real_probs[:, -1] + epsilon))
        D_L_unsupervised2U = -1 * torch.mean(torch.log(D_fake_probs[:, -1] + epsilon))
        d_loss = D_L_Supervised + D_L_unsupervised1U + D_L_unsupervised2U

        #---------------------------------
        #  OPTIMIZATION
        #---------------------------------
        # Avoid gradient accumulation
        gen_optimizer.zero_grad()
        dis_optimizer.zero_grad()

        # Calculate weigth updates
        # retain_graph=True is required since the underlying graph will be deleted after backward
        g_loss.backward(retain_graph=True)
        d_loss.backward() 
        
        # Apply modifications
        gen_optimizer.step()
        dis_optimizer.step()

        # A detail log of the individual losses
        #print("{0:.4f}\t{1:.4f}\t{2:.4f}\t{3:.4f}\t{4:.4f}".
        #      format(D_L_Supervised, D_L_unsupervised1U, D_L_unsupervised2U,
        #            g_loss_d, g_feat_reg))
        training_st1.append(
        {
            'examples': exa,
            'D_L_Supervised': D_L_Supervised,
            'D_L_unsupervised1U': D_L_unsupervised1U,
            'D_L_unsupervised2U': D_L_unsupervised2U,
            'g_loss_d': g_loss_d,
            'g_feat_reg': g_feat_reg,
            'D_L': D_L_Supervised + D_L_unsupervised1U + D_L_unsupervised2U,
            'g_loss': g_loss_d + g_feat_reg
        })
        
        # Save the losses to print them later
        tr_g_loss += g_loss.item()
        tr_d_loss += d_loss.item()

        # Update the learning rate with the scheduler
        if apply_scheduler:
          scheduler_d.step()
          scheduler_g.step()

    # Calculate the average loss over all of the batches.
    avg_train_loss_g = tr_g_loss / len(train_dataloader)
    avg_train_loss_d = tr_d_loss / len(train_dataloader)             
    
    # Measure how long this epoch took.
    training_time = format_time(time.time() - t0)

    print("")
    print("  Average training loss generetor: {0:.4f}".format(avg_train_loss_g))
    print("  Average training loss discriminator: {0:.4f}".format(avg_train_loss_d))
    print("  Training epcoh took: {:}".format(training_time))
        
    # ========================================
    #     TEST ON THE EVALUATION DATASET
    # ========================================
    # After the completion of each training epoch, measure our performance on
    # our test set.
    print("")
    print("Running Test...")

    t0 = time.time()

    # Put the model in evaluation mode--the dropout layers behave differently
    # during evaluation.
    transformer.eval() #maybe redundant
    discriminator.eval()
    generator.eval()

    # Tracking variables 
    total_test_accuracy = 0
   
    total_test_loss = 0
    nb_test_steps = 0

    all_preds = []
    all_labels_ids = []

    #loss
    nll_loss = torch.nn.CrossEntropyLoss(ignore_index=-1)

    # Evaluate data for one epoch
    for batch in test_dataloader:
        
        # Unpack this training batch from our dataloader. 
        b_input_ids = batch[0].to(device)
        b_input_mask = batch[1].to(device)
        b_labels = batch[2].to(device)
        
        # Tell pytorch not to bother with constructing the compute graph during
        # the forward pass, since this is only needed for backprop (training).
        with torch.no_grad():        
            model_outputs = transformer(b_input_ids, attention_mask=b_input_mask)
            hidden_states = model_outputs[-1]
            _, logits, probs = discriminator(hidden_states)
            ###log_probs = F.log_softmax(probs[:,1:], dim=-1)
            #最后一列不取
            filtered_logits = logits[:,0:-1]
            # Accumulate the test loss.
            total_test_loss += nll_loss(filtered_logits, b_labels)
            
        # Accumulate the predictions and the input labels
        _, preds = torch.max(filtered_logits, 1)
        all_preds += preds.detach().cpu()
        all_labels_ids += b_labels.detach().cpu()

    # Report the final accuracy for this validation run.
    all_preds = torch.stack(all_preds).numpy()
    all_labels_ids = torch.stack(all_labels_ids).numpy()
    test_accuracy = np.sum(all_preds == all_labels_ids) / len(all_preds)
    print("  Accuracy: {0:.4f}".format(test_accuracy))
    
    ##########################################################################
    #                                                           #
    test_tp = np.sum(np.multiply(all_labels_ids,all_preds))
    test_fp = np.sum(np.logical_and(np.equal(all_labels_ids, 0), np.equal(all_preds, 1)))
    test_fn = np.sum(np.logical_and(np.equal(all_labels_ids, 1), np.equal(all_preds, 0)))
    test_tn = np.sum(np.logical_and(np.equal(all_labels_ids, 0), np.equal(all_preds, 0)))
    #presion recall F1
    test_p = test_tp/(test_tp + test_fp)
    test_r = test_tp/(test_tp + test_fn)
    test_f1 = np.round(2 * test_p * test_r, 4)/(test_p + test_r)
    test_g = pow((test_p * test_r),0.5)
    mcc_up = test_tp * test_tn - test_tp * test_fn
    mcc_down = pow(((test_tp + test_fp) * (test_tp + test_fn) * (test_tn + test_fp) * (test_tn + test_fn)),0.5)
    test_mcc = mcc_up / mcc_down
    print("  TP:{0:.1f}".format(test_tp))
    print("  FP:{0:.1f}".format(test_fp))
    print("  FN:{0:.1f}".format(test_fn))
    print("  TN:{0:.1f}".format(test_tn))
    
    print("  Precision: {0:.4f}".format(test_p))
    print("  Recall: {0:.4f}".format(test_r))
    print("  F1-score: {0:.4f}".format(test_f1))
    print("  G-score: {0:.4f}".format(test_g))
    print("  MCC: {0:.4f}".format(test_mcc))
    
    #########################################################################
    

    # Calculate the average loss over all of the batches.
    avg_test_loss = total_test_loss / len(test_dataloader)
    avg_test_loss = avg_test_loss.item()
    
    # Measure how long the validation run took.
    test_time = format_time(time.time() - t0)
    
    print("  Test Loss: {0:.4f}".format(avg_test_loss))
    print("  Test took: {:}".format(test_time))

    # Record all statistics from this epoch.
    training_stats.append(
        {
            'epoch': epoch_i + 1,
            'Training Loss generator': avg_train_loss_g,
            'Training Loss discriminator': avg_train_loss_d,
            'Valid. Loss': avg_test_loss,
            'Valid. Accur.': test_accuracy,
            'Valid. Precision': test_p,
            'Valid. Recall': test_r,
            'Valid. F1_score': test_f1,
            'Valid. G_score': test_g,
            'Valid. MCC': test_mcc,
            'Training Time': training_time,
            'Test Time': test_time
        }
    )
    if test_f1 > best_f1:
        best_f1 = test_f1
        es = 0
        torch.save(transformer.state_dict(), "XSGAP0_camel_transformer_weight.pt")
        torch.save(discriminator.state_dict(), "XSGAP0_camel_discriminator_weight.pt")
        torch.save(generator.state_dict(), "XSGAP0_camel_generator_weight.pt")
    else:
        es += 1
        print("Counter {} of 5".format(es))

    if es > 4:
        print("Early stopping with best_f1: ", best_f1, "and val_f1 for this epoch: ", test_f1, "...")
        break





for stat in training_stats:
  print(stat)

print("\nTraining complete!")

print("Total training took {:} (h:mm:ss)".format(format_time(time.time()-total_t0)))





nan = 0
x = []
y1 = []
y2 = []
y3 = []
y4 = []
#training_stats = [{'epoch': 1, 'Training Loss generator': 0.7084165074995585, 'Training Loss discriminator': 1.4781751189913068, 'Valid. Loss': 0.7146640419960022, 'Valid. Accur.': 0.49910873440285203, 'Valid. Precision': 0.49910873440285203, 'Valid. Recall': 1.0, 'Valid. F1_score': 0.6658623067776457, 'Valid. MCC': nan, 'Training Time': '0:00:30', 'Test Time': '0:00:02'},
# {'epoch': 2, 'Training Loss generator': 0.7131793630974633, 'Training Loss discriminator': 1.3904758346932276, 'Valid. Loss': 0.6824660897254944, 'Valid. Accur.': 0.6292335115864528, 'Valid. Precision': 0.652542372881356, 'Valid. Recall': 0.55, 'Valid. F1_score': 0.5969020436927414, 'Valid. MCC': 0.1447150581175037, 'Training Time': '0:00:29', 'Test Time': '0:00:02'}]
for i in range(len(training_stats)):
    x.append(training_stats[i]['epoch'])
    y1.append(training_stats[i]['Valid. Precision'])
    y2.append(training_stats[i]['Valid. Recall'])
    y3.append(training_stats[i]['Valid. F1_score'])
    y4.append(training_stats[i]['Valid. G_score'])
xticks(np.linspace(0,20,21, endpoint = True))
plt.plot(x,y1,'-',color = 'g',marker = 'o', label = "Valid. Precision")
plt.plot(x, y2, '-', color = 'r', marker = 'o', label = "Valid. Recall")
plt.plot(x, y3, '-', color = 'b', marker = 'o', label = "Valid. F1_score")
plt.plot(x, y4, '-', color = 'y', marker = 'o', label = "Valid. G_score")
plt.yticks(np.linspace(0.05,1,20, endpoint = True))
plt.legend(loc = 'best')
plt.grid(True)
plt.xlabel('Epochs',fontsize = 16)
plt.ylabel('Evaluation', fontsize = 16)




