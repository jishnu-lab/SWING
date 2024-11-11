"""
Pretrains a new RoBERTa model (or continues pretraining from a provided checkpoint).
REQUIRES: run_name        - a string; the name of the training run
          train           - a string; the path to training dataset(s)
          eval            - a string; the path to the evaluation dataset(s)
          tokenizer       - a string; the path to the pretrained tokenizer
          max_size        - an integer; the maximum number of tokens in an embedded sequence
          save_path       - a string; the path to the directory in which to save the model checkpoints
          log_steps       - an integer; the number of steps to train before logging
          hidden_size     - an integer; dimension of the hidden layers
          hidden_depth    - an integer; the number of hidden layers in the neural network
          attention_heads - an integer; the number of attention heads in the neural network
          act_func        - a string; the hidden layers' activation function
          hidden_prob     - a float; the hidden layer dropout probability
          att_prob        - a float; the attention layer dropout probability
          class_prob      - a float; the classifier dropout probability
          pos_embed       - a string; the type of positional embedding to use
          epochs          - an integer; the number of epochs to train for
          lr              - a float; the learning rate
          decay           - a float; the weight decay
          per_device      - an integer; the per device batch size
          fp16            - a boolean; whether to use 16-bit floating point precision
          grad_acc        - an integer; the number of gradient steps to accumulate before updating the weights
          no_wandb        - a boolean; whether to turn off logging in wandb
RETURNS:  none
"""

import argparse
import os
import pathlib
import torch
import torch.distributed
import wandb

from datasets         import load_dataset
from transformers     import DataCollatorForLanguageModeling, RobertaConfig, RobertaForMaskedLM, Trainer, TrainingArguments, PreTrainedTokenizerFast, logging

def main():
    logging.set_verbosity_info()

    parser = argparse.ArgumentParser()

    parser.add_argument('--run_name', type = str, help = 'name of model training run')
    parser.add_argument('--train', type = str, help = 'path to training dataset file(s)')
    parser.add_argument('--eval', type = str, help = 'path to training evaluation file(s)')
    parser.add_argument('--tokenizer', type = str, help = 'path to pretrained tokenizer')
    parser.add_argument('--max_size', type = int, help = 'max number of tokens in embedded sequence')
    parser.add_argument('--save_path', type = str, help = 'path to save to', dest = 'model_path')
    parser.add_argument('--project_name', type = str, help = 'wandb project name')
    parser.add_argument('--vocab_size', type = int, help = 'number of tokens in vocabulary', default = 16)
    parser.add_argument('--reset', help = 'restart training from a checkpoint with new arguments', action = 'store_true')

    parser.add_argument('--log_steps', type = int, help = 'number of logging steps', default = 10)
    parser.add_argument('--hidden_size', type = int, help = 'size of hidden layer', dest = 'hid_size', default = 512)
    parser.add_argument('--hidden_depth', type = int, help = 'number of hidden layers', dest = 'hid_depth', default = 8)
    parser.add_argument('--attention_heads', type = int, help = 'number of attention heads', dest = 'att_head', default = 8)
    parser.add_argument('--act_func', type = str, help = 'hidden layer activation function', dest = 'act_func', default = 'gelu')
    parser.add_argument('--hidden_prob', type = float, help = 'hidden layer dropout probability', dest = 'hid_prob', default = 0.1)
    parser.add_argument('--att_prob', type = float, help = 'attention layer dropout probability', dest = 'att_prob', default = 0.1)
    parser.add_argument('--class_prob', type = float, help = 'classifier dropout probability', dest = 'cls_prob', default = 0.1)
    parser.add_argument('--intermed_size', type = int, help = 'size of intermediate layer (feed-forward)', dest = 'int_size', default = 3072)
    parser.add_argument('--pos_embed', type = str, help = 'positional embedding type', dest = 'pos_emb', default = 'relative_key')
    parser.add_argument('--epochs', type = int, help = 'training epochs', dest = 'epochs', default = 5)
    parser.add_argument('--lr', type = float, help = 'learning rate', dest = 'lr', default = 1e-4)
    parser.add_argument('--decay', type = float, help = 'weight decay', dest = 'decay', default = 0.1)
    parser.add_argument('--per_device', type = int, help = 'per device batch size', dest = 'per_dev', default = 16)

    parser.add_argument('--fp16', help = 'use 16-bit floating point precision', action = 'store_true')
    parser.add_argument('--grad_acc', type = int, help = 'number of gradient accumulation steps', default = 1)

    parser.add_argument('--no_wandb', help = 'do not use wandb to record model training', action = 'store_true')

    args = parser.parse_args()

    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    if args.no_wandb:
        os.environ["WANDB_DISABLED"] = "true"
    else:
        wandb.init(project = args.project_name,
                   name = args.run_name)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    ## load pre-trained tokenizer
    tokenizer = PreTrainedTokenizerFast(tokenizer_file = args.tokenizer,
                                        padding_side = 'right',
                                        truncation_side = 'right',
                                        pad_token = '<pad>',
                                        bos_token = '<s>',
                                        eos_token = '</s>',
                                        unk_token = '<unk>',
                                        cls_token = '<s>',
                                        sep_token = '</s>',
                                        mask_token = '<mask>',
                                        additional_special_tokens = ['<|endoftext|>'])

    ## define the data collator
    data_collator = DataCollatorForLanguageModeling(tokenizer       = tokenizer, 
                                                    mlm             = True, 
                                                    mlm_probability = 0.15)

    ## define the training arguments
    training_args = TrainingArguments(output_dir                  = args.model_path,
                                      overwrite_output_dir        = True,
                                      evaluation_strategy         = 'epoch',
                                      save_strategy               = 'epoch',
                                      fp16                        = args.fp16,
                                      gradient_accumulation_steps = args.grad_acc,
                                      log_level                   = 'info',
                                      logging_strategy            = 'epoch',
                                      num_train_epochs            = args.epochs,
                                      learning_rate               = args.lr,
                                      weight_decay                = args.decay,
                                      per_device_train_batch_size = args.per_dev,
                                      per_device_eval_batch_size  = args.per_dev,
                                      logging_steps               = args.log_steps,
                                      run_name                    = args.run_name,
                                      logging_dir                 = os.path.join(args.model_path, 'logs'),
                                      report_to                   = 'wandb')


    ## load datasets
    train_files = [str(x) for x in pathlib.Path(args.train).glob('*.arrow')]
    eval_files = [str(x) for x in pathlib.Path(args.eval).glob('*.arrow')]
   
    train_dataset = load_dataset("arrow", data_files = train_files)['train']
    eval_dataset = load_dataset('arrow', data_files = eval_files)['train']


    # Search for valid checkpoints in output directory
    dirs = [subdir.name for subdir in os.scandir(args.model_path) if subdir.name != 'logs']
    if len(dirs) > 0:
        checkpoints = [int(checkpoint.replace('checkpoint-', '')) for checkpoint in filter(lambda x: 'checkpoint-' in x, dirs)]
        last_checkpoint_num = int(max(checkpoints))
        last_checkpoint_path = os.path.join(args.model_path, 'checkpoint-' + str(last_checkpoint_num))
        model = RobertaForMaskedLM.from_pretrained(last_checkpoint_path)
        trainer = Trainer(model         = model,
                          args          = training_args,
                          data_collator = data_collator,
                          train_dataset = train_dataset,
                          eval_dataset  = eval_dataset)
        if args.reset:
            print("Running with reset arguments")
            trainer.train()
        else:
            print("Running from last checkpoint")
            trainer = Trainer(model         = model,
                              args          = training_args,
                              data_collator = data_collator,
                              train_dataset = train_dataset,
                              eval_dataset  = eval_dataset)
            trainer.train(resume_from_checkpoint = last_checkpoint_path)
    else: 
        # Create the trainer for our model
        trainer = Trainer(model         = model,
                          args          = training_args,
                          data_collator = data_collator,
                          train_dataset = train_dataset,
                          eval_dataset  = eval_dataset)
        ## build model from scratch
        config = RobertaConfig(vocab_size              = args.vocab_size,
                               max_position_embeddings = args.max_size,
                               hidden_size             = args.hid_size,
                               intermediate_size       = args.int_size,
                               num_hidden_layers       = args.hid_depth,
                               num_attention_heads     = args.att_head,
                               hidden_act              = args.act_func,
                               hidden_dropout_prob     = args.hid_prob,
                               attention_dropout_prob  = args.att_prob,
                               classifier_dropout      = args.cls_prob,
                               position_embedding_type = args.pos_emb)
        model = RobertaForMaskedLM(config)
        # Train the model
        trainer.train()

    trainer.save_model(args.model_path + "/SWING_ROBERTA_final")


if __name__ == "__main__":
    main()
